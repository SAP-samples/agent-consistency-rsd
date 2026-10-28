"""Train the RSD metric with NLI costs and n-step TD targets (Algorithm 1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from glob import glob
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import backoff
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AzureOpenAI,
    InternalServerError,
    RateLimitError,
)
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

load_dotenv()

logger = logging.getLogger(__name__)

METRIC_VERSION = "paper_v2"
ANGLE_MODES = ("normalized_atan2", "normalized_acos_clipped", "radians_acos_clipped")
NLI_MODELS = ("gpt-4.1", "deberta", "roberta", "llm")


@dataclass
class Config:
    """Hyperparameters and paths for RSD training."""

    input_dim: int = 1536
    latent_dim: int = 256
    mico_start_dim: int = 512
    beta: float = 5.0  # Angle weight.
    alpha: float = 0.5  # Squared-norm weight.
    norm_reduction: str = "sum"
    angle_mode: str = "normalized_atan2"
    agent_model: str = "gpt-4.1"
    dataset_name: str = "toolsandbox"
    force_recompute: bool = False
    mode: str = "train"

    use_nli: bool = True
    nli_model: str = "gpt-4.1"

    nli_pairs_per_request: int = 1  # >1 enables separately namespaced JSON batching
    nli_workers: int = 5
    show_progress: bool = True

    embeddings_batch_size: int = 64
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    gamma: float = 0.8
    tau: float = 0.005
    n_steps: int = 3
    epochs: int = 300
    model_name: str = "embed-v-4-0"
    seed: int = 42

    resume_checkpoint: str = ""
    loss_fn: str = "mse"  # Equation 8; "huber" is also supported.

    # Derived paths (computed in __post_init__)
    embedding_cache_path: str = ""
    nli_cache_path: str = ""
    save_path: str = ""
    trajectories_path: str = ""
    distance_scores_path: str = ""

    def __post_init__(self):
        if self.nli_model not in NLI_MODELS:
            raise ValueError(f"nli_model must be one of {NLI_MODELS}")
        if self.nli_model == "gpt-4.1":
            self.nli_model = "llm"  # Preserve existing GPT-4.1 caches/checkpoints.
        if self.norm_reduction not in ("sum", "mean"):
            raise ValueError("norm_reduction must be sum or mean")
        if self.angle_mode not in ANGLE_MODES:
            raise ValueError(f"angle_mode must be one of {ANGLE_MODES}")
        if self.nli_pairs_per_request < 1 or self.nli_workers < 1:
            raise ValueError("NLI batch size and workers must be positive")
        directory = f"{self.dataset_name}/{self.agent_model}"
        run_name = (
            f"gamma={self.gamma}_beta={self.beta}_dim={self.latent_dim}"
            f"_n_steps={self.n_steps}_nli_{self.nli_model}_{METRIC_VERSION}"
        )
        if self.norm_reduction == "mean":
            run_name += "_norm=mean"
        if self.angle_mode != "normalized_atan2":
            run_name += f"_angle={self.angle_mode}"
        self.embedding_cache_path = f"./embeddings/{directory}/cached_transitions_{self.mode}_{METRIC_VERSION}.pt"
        self.nli_cache_path = (
            f"./embeddings/{directory}/nli_cache_{self.nli_model}_{METRIC_VERSION}.json"
        )
        self.save_path = f"./models/{directory}/checkpoints_{run_name}/rsd_final.pth"
        self.trajectories_path = f"./traces/{directory}/"
        self.distance_scores_path = f"./results/{directory}/{run_name}_nli_{self.nli_model}/distance_scores.json"

    def metric_metadata(self) -> dict:
        """Identify the metric definition stored with a checkpoint."""
        return {
            "version": METRIC_VERSION,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "nli_model": self.nli_model,
            "encoder": self.model_name,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "norm_reduction": self.norm_reduction,
            "angle_mode": self.angle_mode,
        }

    def validate_checkpoint(self, checkpoint: dict) -> None:
        metric = checkpoint.get("metric", {}).copy()
        if metric.get("version") == METRIC_VERSION:
            # These options were implicit in the first paper_v2 checkpoints.
            metric.setdefault("norm_reduction", "sum")
            metric.setdefault("angle_mode", "normalized_atan2")
        if metric != self.metric_metadata():
            raise ValueError(
                "Checkpoint metric/configuration differs; retrain with the current definition"
            )


def add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared options for training, inference, and analysis."""
    parser.add_argument(
        "--nli-model",
        choices=NLI_MODELS,
        default="gpt-4.1",
        help="NLI judge (default: GPT-4.1 via Azure); llm is an alias for gpt-4.1.",
    )
    parser.add_argument(
        "--norm-reduction",
        choices=("sum", "mean"),
        default="sum",
        help="Squared-norm reduction (default: sum); mean divides by latent dimension.",
    )
    parser.add_argument(
        "--angle-mode",
        choices=ANGLE_MODES,
        default="normalized_atan2",
        help="Angular distance (default: normalized atan2, with zero self-angle).",
    )


def metric_config_from_args(args: argparse.Namespace, **overrides) -> Config:
    return Config(
        nli_model=args.nli_model,
        norm_reduction=args.norm_reduction,
        angle_mode=args.angle_mode,
        **overrides,
    )


# Shared NLI prompt used for d_NLI computation (Definition 1).
NLI_SYSTEM_PROMPT = """You are an expert at comparing LLM agent outputs to determine if they are functionally equivalent.

DEFINITIONS for agent trajectory comparison:

ENTAILMENT (same action/intent): The two outputs are functionally equivalent — they perform the same action or convey the same information, even if worded differently. Specific rules:
- TOOL CALLS: Same function name with same (or equivalent) arguments = ENTAILMENT, regardless of surrounding text
- TEXT RESPONSES: Same answer, same decision, same information conveyed = ENTAILMENT, even with different phrasing
- REQUESTS: Both ask the user for the same information = ENTAILMENT (e.g., both ask for reservation ID)

CONTRADICTION (different action/intent): The two outputs take meaningfully different actions or provide different information. Specific rules:
- TOOL CALLS: Different function names = CONTRADICTION. Same function but different arguments = CONTRADICTION
- TEXT RESPONSES: Different answers, different numbers/prices, different recommendations = CONTRADICTION
- MIXED: One calls a tool while the other sends text to the user = CONTRADICTION

NEUTRAL: One output is a system error, timeout, or transfer-to-human while the other is a normal response.

Respond with ONLY one word: 'entailment', 'contradiction', or 'neutral'."""


class NLIRewardModel:
    """Computes d_NLI rewards using Natural Language Inference (Definition 1, Eq. 5)."""

    def __init__(self, config: Config, output_texts: List[str]):
        self.config = config
        self.output_texts = output_texts
        self.cache = {}
        self.nli_model = config.nli_model
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        namespace = (METRIC_VERSION, self.nli_model, self.deployment, NLI_SYSTEM_PROMPT)
        if config.nli_pairs_per_request > 1:
            namespace += ("json_bidirectional_batch_v1", config.nli_pairs_per_request)
        self.cache_namespace = hashlib.sha256(
            json.dumps(namespace).encode()
        ).hexdigest()
        self.text_hashes = [
            hashlib.sha256(t.encode()).hexdigest() for t in output_texts
        ]

        if os.path.exists(config.nli_cache_path):
            with open(config.nli_cache_path, "r") as f:
                self.cache = json.load(f)
            logger.info(f"Loaded {len(self.cache)} cached NLI results")

        if self.nli_model == "roberta":
            self._init_roberta()
        elif self.nli_model == "deberta":
            self._init_deberta()
        elif self.nli_model == "llm":
            self._init_llm()
        else:
            raise ValueError(f"Unknown NLI model: {self.nli_model}")

    def _init_roberta(self):
        """Initialize RoBERTa-large-MNLI model."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading RoBERTa-large-MNLI model...")
        model_name = "roberta-large-mnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # RoBERTa uses 0=contradiction, 1=neutral, 2=entailment.
        self.label_mapping = {0: 0, 1: 1, 2: 2}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"RoBERTa model loaded on {self.device}")

    def _init_deberta(self):
        """Initialize DeBERTa-v3-large NLI model."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading DeBERTa-v3-large NLI model...")
        model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # DeBERTa reverses the entailment and contradiction labels.
        self.label_mapping = {0: 2, 1: 1, 2: 0}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"DeBERTa-v3-large model loaded on {self.device}")

    def _init_llm(self):
        """Initialize LLM via Azure OpenAI for NLI."""
        logger.info("Initializing LLM for NLI...")
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")

    def _get_cache_key(self, idx_a: int, idx_b: int) -> str:
        """Generate cache key for a pair of outputs."""
        hashes = sorted((self.text_hashes[idx_a], self.text_hashes[idx_b]))
        return f"{self.cache_namespace}:{hashes[0]}:{hashes[1]}"

    @backoff.on_exception(
        backoff.constant,
        (
            RateLimitError,
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            requests.exceptions.Timeout,
        ),
        interval=60,
        max_tries=10,
        jitter=None,
    )
    def _check_entailment_llm_single(self, text_a: str, text_b: str) -> int:
        """Check single-direction entailment using LLM. Returns: 2=entailment, 1=neutral, 0=contradiction."""
        prompt = f"""Compare these two agent outputs:

Output A: {text_a}

Output B: {text_b}"""

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": NLI_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = (
            (response.choices[0].message.content or "").lower().strip().strip(".\"'")
        )
        labels = {"entailment": 2, "neutral": 1, "contradiction": 0}
        if content not in labels:
            raise ValueError(f"Invalid NLI label: {content!r}")
        return labels[content]

    def _check_entailment_llm(self, text_a: str, text_b: str) -> int:
        """Check bidirectional entailment using LLM (Definition 1).

        Returns: 2=entailment (both directions), 1=neutral, 0=contradiction (either direction).
        """
        score_forward = self._check_entailment_llm_single(text_a, text_b)
        if score_forward == 0:
            return 0
        score_backward = self._check_entailment_llm_single(text_b, text_a)

        if score_forward == 2 and score_backward == 2:
            return 2
        elif score_forward == 0 or score_backward == 0:
            return 0
        else:
            return 1

    @backoff.on_exception(
        backoff.constant,
        (
            RateLimitError,
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            requests.exceptions.Timeout,
        ),
        interval=60,
        max_tries=10,
        jitter=None,
    )
    def _check_entailment_llm_batch(
        self, pairs: List[Tuple[int, int, str, str]]
    ) -> Dict[str, int]:
        """Classify output pairs concurrently and return labels keyed by content."""
        if not pairs:
            return {}

        results = {}

        def _process_pair(pair):
            idx_a, idx_b, text_a, text_b = pair
            cache_key = self._get_cache_key(idx_a, idx_b)
            score = self._check_entailment_llm(text_a, text_b)
            return cache_key, score

        if self.config.nli_pairs_per_request > 1:
            chunks = [
                pairs[i : i + self.config.nli_pairs_per_request]
                for i in range(0, len(pairs), self.config.nli_pairs_per_request)
            ]
            with ThreadPoolExecutor(max_workers=self.config.nli_workers) as executor:
                futures = [
                    executor.submit(self._check_entailment_llm_json, chunk)
                    for chunk in chunks
                ]
                for future in as_completed(futures):
                    results.update(future.result())
            return results

        with ThreadPoolExecutor(max_workers=self.config.nli_workers) as executor:
            futures = {executor.submit(_process_pair, p): p for p in pairs}
            for future in as_completed(futures):
                cache_key, score = future.result()
                results[cache_key] = score

        return results

    @backoff.on_exception(
        backoff.expo,
        (
            RateLimitError,
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            ValueError,
        ),
        max_tries=6,
        max_value=30,
    )
    def _check_entailment_llm_json(self, pairs):
        """Request both directions for several pairs in one JSON response."""
        queries = {}
        for i, (_, _, a, b) in enumerate(pairs):
            queries[f"{i}f"] = {"Output A": a, "Output B": b}
            queries[f"{i}b"] = {"Output A": b, "Output B": a}
        prompt = NLI_SYSTEM_PROMPT.replace(
            "Respond with ONLY one word: 'entailment', 'contradiction', or 'neutral'.",
            "Evaluate each numbered comparison independently. Return only a JSON object "
            "mapping every comparison ID to 'entailment', 'contradiction', or 'neutral'. "
            "Do not compare outputs belonging to different IDs.",
        )
        response = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(queries, ensure_ascii=False)},
            ],
        )
        labels = json.loads(response.choices[0].message.content or "")
        allowed = {"entailment": 2, "neutral": 1, "contradiction": 0}
        if (
            not isinstance(labels, dict)
            or set(labels) != set(queries)
            or any(v not in allowed for v in labels.values())
        ):
            raise ValueError("Missing or invalid labels in JSON NLI response")
        results = {}
        for i, (a, b, _, _) in enumerate(pairs):
            forward, backward = allowed[labels[f"{i}f"]], allowed[labels[f"{i}b"]]
            results[self._get_cache_key(a, b)] = (
                0 if 0 in (forward, backward) else 2 if forward == backward == 2 else 1
            )
        return results

    def _check_entailment_roberta(self, text_a: str, text_b: str) -> int:
        """Check entailment using local NLI model. Returns: 2=entailment, 1=neutral, 0=contradiction."""

        inputs = self.tokenizer(
            text_a,
            text_b,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        label = torch.argmax(probs, dim=1).item()
        return self.label_mapping[label]

    def _check_entailment_roberta_bidirectional(self, text_a: str, text_b: str) -> int:
        """Check bidirectional entailment using local NLI model.
        Returns: 2=entailment (both directions), 1=neutral, 0=contradiction (either direction)
        """

        score_forward = self._check_entailment_roberta(text_a, text_b)

        score_backward = self._check_entailment_roberta(text_b, text_a)

        if score_forward == 2 and score_backward == 2:
            return 2  # entailment (equivalent)

        elif score_forward == 0 or score_backward == 0:
            return 0  # contradiction
        else:
            return 1  # neutral

    def _check_entailment_roberta_batch(
        self, pairs: List[Tuple[int, int, str, str]]
    ) -> Dict[str, int]:
        """Classify both directions with the local NLI model."""
        if not pairs:
            return {}

        # Prepare batch: for each pair, we need both A→B and B→A
        batch_premise = []
        batch_hypothesis = []
        pair_info = []  # Track which pair each batch item belongs to

        for idx_a, idx_b, text_a, text_b in pairs:
            batch_premise.append(text_a)
            batch_hypothesis.append(text_b)
            pair_info.append((idx_a, idx_b, "forward"))

            batch_premise.append(text_b)
            batch_hypothesis.append(text_a)
            pair_info.append((idx_a, idx_b, "backward"))

        inputs = self.tokenizer(
            batch_premise,
            batch_hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1).cpu().numpy()

        pair_scores = {}  # Maps (idx_a, idx_b) to {'forward': score, 'backward': score}

        for i, (idx_a, idx_b, direction) in enumerate(pair_info):
            key = (idx_a, idx_b)
            if key not in pair_scores:
                pair_scores[key] = {}
            pair_scores[key][direction] = self.label_mapping[labels[i]]

        results = {}
        for (idx_a, idx_b), scores in pair_scores.items():
            cache_key = self._get_cache_key(idx_a, idx_b)
            score_forward = scores["forward"]
            score_backward = scores["backward"]

            if score_forward == 2 and score_backward == 2:
                results[cache_key] = 2  # entailment (equivalent)

            elif score_forward == 0 or score_backward == 0:
                results[cache_key] = 0  # contradiction
            else:
                results[cache_key] = 1  # neutral

        return results

    def get_nli_score(self, idx_a: int, idx_b: int) -> int:
        """Get NLI score between two outputs. Returns: 2=entailment, 1=neutral, 0=contradiction."""
        return self.batch_get_nli_scores([(idx_a, idx_b)])[0]

    def batch_get_nli_scores(self, idx_pairs: List[Tuple[int, int]]) -> List[int]:
        """Label each distinct text pair once; preserve duplicates and input order."""
        missing = {}
        keys = []
        for idx_a, idx_b in idx_pairs:
            key = self._get_cache_key(idx_a, idx_b)
            keys.append(key)
            text_a, text_b = self.output_texts[idx_a], self.output_texts[idx_b]
            if text_a == text_b:
                self.cache[key] = 2  # Definition 1: d_NLI(a, a) = 0.
            elif key not in self.cache:
                missing[key] = (idx_a, idx_b, text_a, text_b)
        if missing:
            classify = (
                self._check_entailment_roberta_batch
                if self.nli_model in ("roberta", "deberta")
                else self._check_entailment_llm_batch
            )
            computed = classify(list(missing.values()))
            if set(computed) != set(missing) or any(
                v not in (0, 1, 2) for v in computed.values()
            ):
                raise ValueError("NLI batch returned missing or invalid labels")
            self.cache.update(computed)
        scores = [self.cache[key] for key in keys]
        if any(score not in (0, 1, 2) for score in scores):
            raise ValueError("NLI cache contains invalid labels")
        return scores

    def save_cache(self):
        """Save NLI cache to disk."""
        os.makedirs(os.path.dirname(self.config.nli_cache_path), exist_ok=True)
        temporary = self.config.nli_cache_path + ".tmp"
        with open(temporary, "w") as f:
            json.dump(self.cache, f)
        os.replace(temporary, self.config.nli_cache_path)
        logger.info(f"Saved {len(self.cache)} NLI results to cache")


class MICONetwork(nn.Module):
    """MICo distance network φ_θ mapping state embeddings to latent space (Eq. 2)."""

    def __init__(self, config: Config):
        super().__init__()
        self.beta = config.beta
        self.alpha = config.alpha
        self.norm_reduction = config.norm_reduction
        self.angle_mode = config.angle_mode
        self.phi = nn.Sequential(
            nn.Linear(config.input_dim, config.mico_start_dim),
            nn.LayerNorm(config.mico_start_dim),
            nn.ReLU(),
            nn.Linear(config.mico_start_dim, config.mico_start_dim // 2),
            nn.ReLU(),
            nn.Linear(config.mico_start_dim // 2, config.latent_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.phi(x)

    def compute_distance(
        self, phi_x: torch.Tensor, phi_y: torch.Tensor
    ) -> torch.Tensor:
        """Compute MICo distance between two batches of latent representations (Eq. 2).

        Returns distance tensor of shape (batch_size, 1).
        """
        norm_x_sq = self.squared_norm(phi_x)
        norm_y_sq = self.squared_norm(phi_y)
        return self.alpha * (norm_x_sq + norm_y_sq) + self.beta * self.angular_distance(
            phi_x, phi_y
        )

    def angular_distance(
        self, phi_x: torch.Tensor, phi_y: torch.Tensor
    ) -> torch.Tensor:
        """Normalized angle by default; clipped modes retain a nonzero self-angle."""
        if self.angle_mode != "normalized_atan2":
            x = F.normalize(phi_x, p=2, dim=1)
            y = F.normalize(phi_y, p=2, dim=1)
            cosine = (x * y).sum(dim=1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            angle = torch.acos(cosine)
            return (
                angle / math.pi
                if self.angle_mode == "normalized_acos_clipped"
                else angle
            )
        length_x = torch.linalg.vector_norm(phi_x, dim=1, keepdim=True)
        length_y = torch.linalg.vector_norm(phi_y, dim=1, keepdim=True)
        x = phi_x / torch.where(length_x > 0, length_x, torch.ones_like(length_x))
        y = phi_y / torch.where(length_y > 0, length_y, torch.ones_like(length_y))
        # Equivalent to acos(cosine)/pi for nonzero vectors, but with finite
        # gradients at parallel/antiparallel vectors and exact zero self-angle.
        difference = torch.linalg.vector_norm(x - y, dim=1, keepdim=True)
        total = torch.linalg.vector_norm(x + y, dim=1, keepdim=True)
        both_zero = (length_x == 0) & (length_y == 0)
        total = torch.where(both_zero, torch.ones_like(total), total)
        angle = 2.0 * torch.atan2(difference, total) / math.pi
        # Zero-vector extension: both zero -> 0; exactly one zero -> 1/2.
        return angle

    def squared_norm(self, phi: torch.Tensor) -> torch.Tensor:
        squared = phi.square()
        return (
            squared.sum(dim=1, keepdim=True)
            if self.norm_reduction == "sum"
            else squared.mean(dim=1, keepdim=True)
        )

    def self_distance(self, phi: torch.Tensor) -> torch.Tensor:
        """Evaluate U(s, s), including the angle term when clipping is enabled."""
        if self.angle_mode != "normalized_atan2":
            return self.compute_distance(phi, phi)
        return 2.0 * self.alpha * self.squared_norm(phi)


class TraceLoader:
    """Parses agent trajectory JSON files into state-action transitions."""

    def linearize_json(self, data):
        if isinstance(data, dict):
            return ", ".join(
                [
                    f"{k} is {self.linearize_json(v)}"
                    for k, v in data.items()
                    if v is not None and k != "reasoning"
                ]
            )
        elif isinstance(data, list):
            return "; ".join([self.linearize_json(item) for item in data])
        else:
            return str(data)

    def format_turn(self, turn):
        role = turn["role"]
        content = turn.get("content")
        if role == "user":
            return f"User: {content}"
        elif role == "tool":
            try:
                tool_data = json.loads(content)
                readable_data = self.linearize_json(tool_data)
                return f"System: {readable_data}"
            except Exception:
                return f"System: {content}"
        elif role == "agent":
            if turn.get("tool_calls"):
                calls = []
                for tc in turn["tool_calls"]:
                    func_name = tc["name"]
                    func_args = self.linearize_json(tc["arguments"])
                    calls.append(f"calls tool '{func_name}' with arguments {func_args}")
                return f"Agent: {content + ' ' if content else ''}{', '.join(calls)}"
            else:
                return f"Agent: {content}"

    def load_file(self, file_path):
        """Load transitions with task, run, and turn IDs for aligned pairing.

        Returns list of tuples: (state_str, output_str, next_state_str, done, sample_id, run_id, turn_number)
        """

        filename = os.path.basename(file_path)
        parts = filename.replace(".json", "").split("_")
        run_id = int(parts[1]) if len(parts) >= 3 else 0
        sample_id = int(parts[2]) if len(parts) >= 3 else 0

        with open(file_path, "r") as f:
            raw_traces = json.load(f)
        # Outer lists are consecutive conversation segments of this rollout.
        # Consume every observation once, through the next agent action.
        messages = [message for segment in raw_traces for message in segment]
        transitions = []
        history = []
        i = 0
        while i < len(messages):
            message = messages[i]
            if message["role"] != "agent":
                history.append(self.format_turn(message))
                i += 1
                continue
            state = " [SEP] ".join(history) or "Start of conversation"
            output = self.format_turn(message)
            history.append(output)
            i += 1
            while i < len(messages) and messages[i]["role"] != "agent":
                history.append(self.format_turn(messages[i]))
                i += 1
            transitions.append(
                (
                    state,
                    output,
                    " [SEP] ".join(history),
                    i == len(messages),
                    sample_id,
                    run_id,
                    len(transitions),
                )
            )
        return transitions


class TracePreprocessor:
    """Computes and caches text embeddings for transitions via Cohere embed-v4."""

    def __init__(self, config: Config):
        self.config = config
        self.client = None

    def _load_client(self):
        if self.client is None:
            from cohere_embed_v4 import get_client

            self.client = get_client()

    def _get_embeddings(self, texts: List[str]):
        batch_size = self.config.embeddings_batch_size
        all_embeddings = []
        unique = list(dict.fromkeys(texts))
        for i in tqdm(
            range(0, len(unique), batch_size), disable=not self.config.show_progress
        ):
            batch_texts = unique[i : i + batch_size]
            batch_embeddings = self._get_batch_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)

        by_text = dict(zip(unique, all_embeddings))
        return torch.tensor([by_text[text] for text in texts], dtype=torch.float32)

    @backoff.on_exception(
        backoff.constant,
        (
            RateLimitError,
            InternalServerError,
            APIConnectionError,
            APITimeoutError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
        interval=60,
        max_tries=25,
        jitter=None,
        on_backoff=lambda details: logger.warning(
            f"Backing off {details['wait']}s after {details['tries']} tries..."
        ),
    )
    def _get_batch_embeddings(self, batch_texts: List[str]):
        self._load_client()
        response = self.client.embeddings.create(
            input=batch_texts,
            model=self.config.model_name,
            dimensions=self.config.input_dim,
            encoding_format="float",
            extra_body={"truncate": "START"},
            extra_headers={"extra-parameters": "pass-through"},
        )
        return [item.embedding for item in response.data]

    def process(self, transitions: List[Tuple]):
        """Embed transitions, reusing the cache when content and encoder match.

        Accepts (state, output, next_state, done), optionally followed by
        (sample_id, run_id, turn_number).
        """
        num_samples = len(transitions)
        if not num_samples:
            raise ValueError("No transitions to embed")
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    METRIC_VERSION,
                    self.config.model_name,
                    self.config.input_dim,
                    transitions,
                ],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        next_indices = torch.full((num_samples,), -1, dtype=torch.long)

        has_metadata = len(transitions[0]) == 7 if transitions else False

        for i in range(num_samples - 1):
            done = transitions[i][3]
            same_rollout = (
                not has_metadata or transitions[i][4:6] == transitions[i + 1][4:6]
            )
            if not done and same_rollout:
                if transitions[i][2] != transitions[i + 1][0]:
                    raise ValueError(f"Broken state continuity at transition {i}")
                next_indices[i] = i + 1
            else:
                next_indices[i] = -1

        if has_metadata:
            sample_ids = torch.tensor([t[4] for t in transitions], dtype=torch.long)
            run_ids = torch.tensor([t[5] for t in transitions], dtype=torch.long)
            turn_numbers = torch.tensor([t[6] for t in transitions], dtype=torch.long)
        else:
            # Legacy format: no metadata
            sample_ids = torch.zeros(num_samples, dtype=torch.long)
            run_ids = torch.zeros(num_samples, dtype=torch.long)
            turn_numbers = torch.zeros(num_samples, dtype=torch.long)

        if (
            os.path.exists(self.config.embedding_cache_path)
            and not self.config.force_recompute
        ):
            logger.info(
                f"Loading cached embeddings from {self.config.embedding_cache_path}"
            )
            try:
                # weights_only=False: cache contains string lists alongside tensors
                data = torch.load(self.config.embedding_cache_path, weights_only=False)
                if data.get("fingerprint") == fingerprint and len(
                    data["outputs"]
                ) == len(transitions):
                    data["done"] = torch.tensor(
                        [t[3] for t in transitions], dtype=torch.float32
                    )
                    data["next_indices"] = next_indices

                    data["sample_ids"] = sample_ids
                    data["run_ids"] = run_ids
                    data["turn_numbers"] = turn_numbers
                    return data
                else:
                    logger.warning(
                        "Cached embeddings content/configuration mismatch, recomputing."
                    )
            except Exception as e:
                logger.error(f"Error loading cached embeddings: {e}. Recomputing.")

        logger.info("Computing embeddings for transitions...")
        states = [t[0] for t in transitions]
        outputs = [t[1] for t in transitions]
        next_states = [t[2] for t in transitions]
        combined = self._get_embeddings(states + next_states + outputs)
        z_current, z_next, z_outputs = combined.split(num_samples)

        data = {
            "fingerprint": fingerprint,
            "z_current": z_current,
            "z_next": z_next,
            "z_outputs": z_outputs,
            "outputs": outputs,
            "done": torch.tensor([t[3] for t in transitions], dtype=torch.float32),
            "next_indices": next_indices,
            "sample_ids": sample_ids,
            "run_ids": run_ids,
            "turn_numbers": turn_numbers,
        }

        logger.info(f"Saving embeddings to {self.config.embedding_cache_path}")

        if not os.path.exists(os.path.dirname(self.config.embedding_cache_path)):
            os.makedirs(os.path.dirname(self.config.embedding_cache_path))

        torch.save(data, self.config.embedding_cache_path)
        return data


class RSDDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed transition embeddings."""

    def __init__(self, data_dict):
        self.z_current = data_dict["z_current"]
        self.z_next = data_dict["z_next"]
        self.z_outputs = data_dict["z_outputs"]
        self.outputs = data_dict["outputs"]
        self.done = data_dict["done"]
        self.next_indices = data_dict["next_indices"]

        self.sample_ids = data_dict.get(
            "sample_ids", torch.zeros(len(self.outputs), dtype=torch.long)
        )
        self.run_ids = data_dict.get(
            "run_ids", torch.zeros(len(self.outputs), dtype=torch.long)
        )
        self.turn_numbers = data_dict.get(
            "turn_numbers", torch.zeros(len(self.outputs), dtype=torch.long)
        )

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return {
            "index": idx,
            "z_current": self.z_current[idx],
            "z_next": self.z_next[idx],
            "z_outputs": self.z_outputs[idx],
            "output_text": self.outputs[idx],
            "done": self.done[idx],
            "sample_id": self.sample_ids[idx],
            "run_id": self.run_ids[idx],
            "turn_number": self.turn_numbers[idx],
        }


class RSDTrainer:
    """Trains the MICo network using n-step TD targets (Algorithm 1)."""

    def __init__(self, config: Config, model: MICONetwork, dataset: RSDDataset):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.target_model = MICONetwork(config).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        if self.config.loss_fn == "huber":
            self.loss_fn = nn.HuberLoss(delta=1.0)
        else:
            self.loss_fn = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.full_z_outputs = dataset.z_outputs.to(self.device)
        self.full_z_current = dataset.z_current.to(self.device)
        self.full_z_next = dataset.z_next.to(self.device)
        self.full_dones = dataset.done.to(self.device)
        self.full_next_indices = dataset.next_indices.to(self.device)
        self.output_texts = dataset.outputs

        # Keep pairing metadata on the CPU for Python sampling.
        self.sample_ids = dataset.sample_ids.numpy()
        self.run_ids = dataset.run_ids.numpy()
        self.turn_numbers = dataset.turn_numbers.numpy()

        self.n_steps = self.config.n_steps

        self.task_runs = defaultdict(lambda: defaultdict(dict))
        for idx, (sample, run, turn) in enumerate(
            zip(self.sample_ids, self.run_ids, self.turn_numbers)
        ):
            self.task_runs[int(sample)][int(run)][int(turn)] = idx
        self.task_runs = {
            task: runs for task, runs in self.task_runs.items() if len(runs) >= 2
        }
        if not self.task_runs:
            raise ValueError(
                "Algorithm 1 requires at least two independent rollouts per task"
            )

        self.nli_model = None
        if self.config.use_nli:
            logger.info(f"Initializing NLI model: {self.config.nli_model}")
            self.nli_model = NLIRewardModel(config, self.output_texts)

    def prepare_nli_labels(self) -> int:
        """Label all aligned pairs in the training set before optimization."""
        if self.nli_model is None:
            return 0
        chunk = []
        processed = 0
        last_saved = 0
        try:
            for runs in self.task_runs.values():
                for run_a, run_b in combinations(sorted(runs), 2):
                    for turn in sorted(set(runs[run_a]) & set(runs[run_b])):
                        chunk.append((runs[run_a][turn], runs[run_b][turn]))
                        if len(chunk) == self.config.batch_size:
                            self.nli_model.batch_get_nli_scores(chunk)
                            processed += len(chunk)
                            chunk = []
                            if processed - last_saved >= 256:
                                self.nli_model.save_cache()
                                last_saved = processed
                                logger.info(
                                    f"Prepared NLI labels for {processed} aligned pairs"
                                )
            if chunk:
                self.nli_model.batch_get_nli_scores(chunk)
                processed += len(chunk)
        finally:
            self.nli_model.save_cache()
        logger.info(f"Prepared all {processed} aligned NLI pairs before training")
        return processed

    def soft_update(self):
        for param, target_param in zip(
            self.model.parameters(), self.target_model.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * param.data
                + (1.0 - self.config.tau) * target_param.data
            )

    def sample_aligned_pairs(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Algorithm 1: sample task, two distinct runs, then a shared turn.

        Histories at that turn can differ; these are state-pair targets, not
        observations of U(s,s) unless the histories actually coincide.
        """
        pairs = []
        for _ in range(batch_size):
            runs = self.task_runs[random.choice(list(self.task_runs))]
            run_a, run_b = random.sample(sorted(runs), 2)
            common = sorted(set(runs[run_a]) & set(runs[run_b]))
            if not common:
                raise ValueError("Paired trajectories have no common turn")
            turn = random.choice(common)
            pairs.append((runs[run_a][turn], runs[run_b][turn]))
        a, b = zip(*pairs)
        return (
            torch.tensor(a, device=self.device),
            torch.tensor(b, device=self.device),
        )

    @torch.no_grad()
    def compute_td_target(
        self, idx_a: torch.Tensor, idx_b: torch.Tensor
    ) -> torch.Tensor:
        """Equation 7, truncated at the end of the shorter available rollout.

        The paper does not specify absorbing-action costs after termination;
        only jointly observed actions are compared, with no terminal bootstrap.
        """
        target = torch.zeros(len(idx_a), device=self.device)
        curr_a, curr_b = idx_a.clone(), idx_b.clone()
        active = torch.ones(len(idx_a), dtype=torch.bool, device=self.device)
        for step in range(self.n_steps):
            positions = active.nonzero(as_tuple=True)[0]
            if not len(positions):
                break
            a, b = curr_a[positions], curr_b[positions]
            reward = self.get_proxy_reward(
                self.full_z_outputs[a], self.full_z_outputs[b], a, b
            ).squeeze(1)
            target[positions] += self.config.gamma**step * reward
            next_a, next_b = self.full_next_indices[a], self.full_next_indices[b]
            continuing = (next_a != -1) & (next_b != -1)
            active[positions] = continuing
            positions = positions[continuing]
            curr_a[positions], curr_b[positions] = (
                next_a[continuing],
                next_b[continuing],
            )
        positions = active.nonzero(as_tuple=True)[0]
        if len(positions):
            phi_a = self.target_model(self.full_z_current[curr_a[positions]])
            phi_b = self.target_model(self.full_z_current[curr_b[positions]])
            target[positions] += (
                self.config.gamma** self.n_steps
                * self.target_model.compute_distance(phi_a, phi_b).squeeze(1)
            )
        return target.unsqueeze(1)

    def get_proxy_reward(
        self,
        z_out_a: torch.Tensor,
        z_out_b: torch.Tensor,
        idx_a: Optional[torch.Tensor] = None,
        idx_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute stepwise reward d_NLI between action pairs (Eq. 5).

        When use_nli=True, uses an NLI classifier to label pairs as
        entailment (0.0), neutral (0.5), or contradiction (1.0).
        Falls back to cosine-similarity proxy when NLI is disabled.
        """
        if self.config.use_nli and self.nli_model is not None:
            batch_size = z_out_a.shape[0]
            rewards = torch.zeros(
                batch_size, dtype=torch.float32, device=z_out_a.device
            )

            idx_pairs = [(idx_a[i].item(), idx_b[i].item()) for i in range(batch_size)]

            nli_scores = self.nli_model.batch_get_nli_scores(idx_pairs)

            for i, nli_score in enumerate(nli_scores):
                if nli_score == 2:  # entailment
                    rewards[i] = 0.0
                elif nli_score == 1:  # neutral
                    rewards[i] = 0.5
                elif nli_score == 0:
                    rewards[i] = 1.0
                else:
                    raise ValueError(f"Invalid NLI score: {nli_score!r}")

            return rewards.unsqueeze(1)
        else:
            sim_score = F.cosine_similarity(z_out_a, z_out_b, dim=1, eps=1e-8)
            dist = 1.0 - sim_score
            scale = 20.0
            shift = 0.15
            r_proxy = torch.sigmoid(scale * (dist - shift))

            return r_proxy.unsqueeze(1)

    def train_epoch(self, dataloader):
        """Train on task-uniform, time-aligned independent rollout pairs."""
        self.model.train()
        total_loss = 0.0
        steps = 0
        for batch in tqdm(dataloader, disable=not self.config.show_progress):
            idx_a, idx_b = self.sample_aligned_pairs(len(batch["index"]))
            target = self.compute_td_target(idx_a, idx_b)
            phi_a = self.model(self.full_z_current[idx_a])
            phi_b = self.model(self.full_z_current[idx_b])
            loss = self.loss_fn(self.model.compute_distance(phi_a, phi_b), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite TD loss")
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.soft_update()
            total_loss += loss.item()
            steps += 1
        if not steps:
            raise ValueError("Empty training loader")
        avg_loss = total_loss / steps
        self.scheduler.step(avg_loss)
        return avg_loss

    def metric_metadata(self):
        return self.config.metric_metadata()

    def save_model(self, path):
        torch.save(
            {
                "metric": self.metric_metadata(),
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def save_checkpoint(self, path, epoch):
        checkpoint = {
            "metric": self.metric_metadata(),
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.config.validate_checkpoint(checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "target_model_state_dict" in checkpoint:
            self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
        else:
            self.target_model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = 0
            logger.warning(
                "Weights-only checkpoint: starting with fresh optimizer/scheduler"
            )
        logger.info(f"Resumed from checkpoint: {path} (epoch {start_epoch})")
        return start_epoch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_train_test_split(
    files: list[str], seed: int = 42, train_ratio: float = 0.8
) -> tuple[list[str], list[str]]:
    """Split by task ID, keeping every run of a task in the same partition."""
    sample_ids = set()
    for f in files:
        filename = os.path.basename(f)
        parts = filename.replace(".json", "").split("_")
        if len(parts) >= 3:
            sample_ids.add(parts[2])

    sample_ids = sorted(list(sample_ids))
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    train_size = int(train_ratio * len(sample_ids))
    train_sample_ids = set(sample_ids[:train_size])
    test_sample_ids = set(sample_ids[train_size:])

    train_files, test_files = [], []
    for f in files:
        filename = os.path.basename(f)
        parts = filename.replace(".json", "").split("_")
        if len(parts) >= 3:
            sid = parts[2]
            if sid in train_sample_ids:
                train_files.append(f)
            elif sid in test_sample_ids:
                test_files.append(f)

    return train_files, test_files


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_metric_arguments(parser)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    config = metric_config_from_args(args, epochs=args.epochs)
    set_seed(config.seed)

    logger.info("--- Configuration ---")
    logger.info(f"Latent Dim: {config.latent_dim}")
    logger.info(f"Beta: {config.beta}")
    logger.info(f"Norm reduction: {config.norm_reduction}")
    logger.info(f"Angle mode: {config.angle_mode}")
    logger.info(
        f"NLI model: {'gpt-4.1 (Azure)' if config.nli_model == 'llm' else config.nli_model}"
    )
    logger.info(f"Loss Function: {config.loss_fn}")
    logger.info("Pairing: same task, distinct runs, aligned turn (Algorithm 1)")

    loader = TraceLoader()

    files = sorted(glob(config.trajectories_path + "*.json"))
    train_files, _ = get_train_test_split(files, config.seed)
    logger.info(f"Total files: {len(files)}, Training: {len(train_files)}")

    all_transitions = []
    for f_path in train_files:
        if os.path.exists(f_path):
            all_transitions.extend(loader.load_file(f_path))
        else:
            logger.warning(f"File {f_path} does not exist, skipping.")

    if not all_transitions:
        logger.error("No transitions found. Exiting.")
        sys.exit(1)

    logger.info(f"Total transitions loaded: {len(all_transitions)}")

    preprocessor = TracePreprocessor(config)
    processed_data = preprocessor.process(all_transitions)

    dataset = RSDDataset(processed_data)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = MICONetwork(config)
    trainer = RSDTrainer(config, model, dataset)

    start_epoch = 0
    if config.resume_checkpoint:
        start_epoch = trainer.load_checkpoint(config.resume_checkpoint)

    # Algorithm 1 prepares all action-pair costs before the optimization loop.
    trainer.prepare_nli_labels()

    logger.info(f"Starting training from epoch {start_epoch + 1}")
    for epoch in range(start_epoch, config.epochs):
        start_time = time.time()
        avg_loss = trainer.train_epoch(dataloader)
        duration = time.time() - start_time
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.6f}, Time: {duration:.2f}s"
        )

        checkpoint_dir = os.path.dirname(config.save_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        base_name = os.path.splitext(os.path.basename(config.save_path))[0]
        epoch_path = os.path.join(checkpoint_dir, f"{base_name}_epoch={epoch + 1}.pth")
        trainer.save_checkpoint(epoch_path, epoch + 1)

        # Save NLI cache every epoch to avoid losing results on crash
        if config.use_nli and trainer.nli_model is not None:
            trainer.nli_model.save_cache()

    if config.use_nli and trainer.nli_model is not None:
        trainer.nli_model.save_cache()

    trainer.save_model(config.save_path)


if __name__ == "__main__":
    main()
