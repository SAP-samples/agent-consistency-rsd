"""Extract agent trajectories from raw benchmark logs into the trace format expected by train.py."""

from __future__ import annotations

import ast
import json
import logging
import os
import pickle
from glob import glob

logger = logging.getLogger(__name__)


def convert_tool_to_tau_trace(tool_trace_data):
    """Convert ToolSandbox messages and tool calls to the shared trace schema."""
    tau_trace_data = []

    for turn in tool_trace_data:
        converted_turn = []
        for message in turn:
            new_message = message.copy()

            if new_message.get("role") == "user":
                new_message.pop("user_details", None)
            elif new_message.get("role") == "assistant":
                new_message["role"] = "agent"

                if "tool_calls" in new_message and new_message["tool_calls"]:
                    converted_tool_calls = []
                    for call in new_message["tool_calls"]:
                        new_call = {
                            "id": call.get("id"),
                            "name": call.get("function", {}).get("name"),
                            "arguments": {},
                        }

                        args_str = call.get("function", {}).get("arguments")
                        if isinstance(args_str, str):
                            try:
                                new_call["arguments"] = json.loads(args_str)
                            except json.JSONDecodeError:
                                new_call["arguments"] = args_str
                        else:
                            new_call["arguments"] = args_str

                        converted_tool_calls.append(new_call)
                    new_message["tool_calls"] = converted_tool_calls
                else:
                    new_message["tool_calls"] = None

            elif new_message.get("role") == "tool":
                if "tool_call_id" in new_message:
                    new_message["tool_id"] = new_message.pop("tool_call_id")
                    new_message["content"] = message.get("content", "")

                new_message.pop("name", None)

            converted_turn.append(new_message)
        tau_trace_data.append(converted_turn)

    return tau_trace_data


def main():
    logging.basicConfig(level=logging.INFO)

    RAW_LOGS_DIR = "./data/raw_logs/tau2bench/retail/gpt-4.1/expert/"  # Update to your raw logs directory
    files = glob(os.path.join(RAW_LOGS_DIR, "*.json"))
    files = sorted(files)

    save_dir = "./traces/tau2bench_retail/gpt-4.1_temp=0/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    reward_dict = {}

    for i, file in enumerate(files):
        with open(file, "r") as f:
            logs = json.load(f)

        samples = logs["samples"]

        for idx, sample in enumerate(samples):
            traj = json.loads(sample["output"]["choices"][0]["message"]["content"])[
                "trajectories"
            ]

            for turn in traj:
                turn[:] = [step for step in turn if step.get("role") != "token_probs"]

            reward_arr = ast.literal_eval(
                sample["scores"]["model_graded_fact_subgoals_auc"]["value"][
                    "progress_turns"
                ]
            )

            progress_rate = reward_arr[-1]

            reward_dict[f"trajectories_{i}_{idx}"] = progress_rate

            with open(f"{save_dir}/trajectories_{i}_{idx}.json", "w") as out_f:
                json.dump(traj, out_f, indent=2)

            with open(f"{save_dir}/reward_dict.pkl", "wb") as f:
                pickle.dump(reward_dict, f)


if __name__ == "__main__":
    main()
