"""Score held-out trajectories with the trained RSD metric (Algorithm 2)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from glob import glob

import torch

from train import (
    Config,
    MICONetwork,
    RSDDataset,
    TraceLoader,
    TracePreprocessor,
    add_metric_arguments,
    get_train_test_split,
    metric_config_from_args,
    set_seed,
)

logger = logging.getLogger(__name__)


def load_model(config: Config) -> MICONetwork:
    """Load a trained MICo network from the checkpoint path in config."""
    model = MICONetwork(config)
    checkpoint = torch.load(config.save_path, map_location="cpu", weights_only=True)
    config.validate_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_metric_arguments(parser)
    args = parser.parse_args()
    config = metric_config_from_args(args, mode="test")

    model = load_model(config)

    results = []

    set_seed(config.seed)

    logger.info("--- Configuration ---")
    logger.info(f"Latent Dim: {config.latent_dim}")
    logger.info(f"Beta: {config.beta}")
    logger.info(f"Norm reduction: {config.norm_reduction}")
    logger.info(f"Trajectories Path: {config.trajectories_path}")

    loader = TraceLoader()

    files = sorted(glob(config.trajectories_path + "*.json"))
    _, test_files = get_train_test_split(files, config.seed)
    logger.info(f"Total files: {len(files)}, Test: {len(test_files)}")

    reward_dict_path = config.trajectories_path + "reward_dict.pkl"
    progress_rates = {}
    if os.path.exists(reward_dict_path):
        with open(reward_dict_path, "rb") as f:
            progress_rates = pickle.load(f)

    all_test_transitions = []
    file_transition_ranges = []  # Store (start_idx, end_idx, file_path) for each file

    for trace in test_files:
        transitions = loader.load_file(trace)
        if len(transitions) == 0:
            continue
        start_idx = len(all_test_transitions)
        all_test_transitions.extend(transitions)
        end_idx = len(all_test_transitions)
        file_transition_ranges.append((start_idx, end_idx, trace))

    logger.info(f"Total test transitions: {len(all_test_transitions)}")

    preprocessor = TracePreprocessor(config)
    processed_data = preprocessor.process(all_test_transitions)

    dataset = RSDDataset(processed_data)

    logger.info(f"Total dataset size: {len(dataset)}")

    all_scores = []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            z_current = item["z_current"]
            phi_start = model(z_current.unsqueeze(0))
            u_start_start = model.self_distance(phi_start)
            all_scores.append(u_start_start.item())

    for start_idx, end_idx, trace in file_transition_ranges:
        trace_scores = all_scores[start_idx:end_idx]
        trace_result = {
            "trace_path": trace,
            "scores": trace_scores,
        }
        # Success labels are optional annotations, not inputs to Algorithm 2.
        trace_key = os.path.splitext(os.path.basename(trace))[0]
        if trace_key in progress_rates:
            trace_result["progress_rate"] = progress_rates[trace_key]
        results.append(trace_result)

    output_file = config.distance_scores_path

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()
