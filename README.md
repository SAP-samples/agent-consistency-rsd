# Agent Consistency

## Recursive Semantic Divergence for LLM Agent Consistency (NeurIPS 2026)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/agent-consistency-rsd)](https://api.reuse.software/info/github.com/SAP-samples/agent-consistency-rsd)

Paper Link: https://arxiv.org/abs/2610.33012

## Overview

RSD measures the expected discounted semantic disagreement between independent agent rollouts from the same state. An NLI classifier labels each action pair; a MICo network learns the trajectory-level distance with temporal-difference updates.

## Installation

```bash
git clone https://github.com/SAP-samples/agent-consistency-rsd
cd agent-consistency-rsd
pip install -r requirements.txt
cp .env.example .env
```

Set Azure OpenAI credentials in `.env`:

```dotenv
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
```

Embeddings use Cohere embed-v4 through Azure. GPT-4.1 NLI uses the same deployment. DeBERTa and RoBERTa run locally via Transformers.

## Data

Place traces in `traces/<dataset>/<model>/` as `trajectories_<run_id>_<sample_id>.json`. Each file contains conversation segments with `role` (`user`, `agent`, `tool`) and `content` fields. Training requires at least two independent runs per task. The train/test split holds out 20% of task IDs.

An optional `reward_dict.pkl` maps trace filenames (without `.json`) to progress rates for downstream analysis.

To convert raw benchmark logs, set `RAW_LOGS_DIR` and `save_dir` in `extract_trace.py` and run:

```bash
python extract_trace.py
```

## Usage

Set `dataset_name` and `agent_model` in `Config` in `train.py`. Defaults are `toolsandbox` and `gpt-4.1`.

```bash
python train.py --epochs 300
python test.py
python analyze_distances.py
```

Shared metric options across all commands:

| Option | Default | Choices |
| --- | --- | --- |
| `--nli-model` | `gpt-4.1` | `gpt-4.1`, `deberta`, `roberta`, `llm` |
| `--norm-reduction` | `sum` | `sum`, `mean` |
| `--angle-mode` | `normalized_atan2` | `normalized_atan2`, `normalized_acos_clipped`, `radians_acos_clipped` |

Use matching options for training and inference. Each variant has a separate checkpoint, and loading validates the metric configuration.


## Files

| File | Purpose |
| --- | --- |
| `train.py` | Configuration, trace loading, embeddings, NLI, and training (Algorithm 1) |
| `test.py` | Held-out inference (Algorithm 2) |
| `analyze_distances.py` | Score analysis, plots, and statistical tests |
| `cohere_embed_v4.py` | Azure embedding client |
| `extract_trace.py` | Raw-log conversion |

## Known Issues

No known issues.

## How to Obtain Support

[Create an issue](https://github.com/SAP-samples/agent-consistency-rsd/issues) in this repository if you find a bug or have questions about the content.

For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing

If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## Citation

```bibtex
@misc{abichandani2026recursivesemanticdivergencellm,
      title={Recursive Semantic Divergence for LLM Agent Consistency}, 
      author={Harshavardhan Abichandani and Penny Chong and Atin Ghosh and Daniel Dahlmeier},
      year={2026},
      eprint={2610.33012},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.33012}, 
}
```

## License

Copyright 2026 SAP SE or an SAP affiliate company and agent-consistency-rsd contributors. Please see our [LICENSE](LICENSE) for copyright and license information. Detailed information including third-party components and their licensing/copyright information is available [via the REUSE tool](https://api.reuse.software/info/github.com/SAP-samples/agent-consistency-rsd).
