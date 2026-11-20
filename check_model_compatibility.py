"""Identify supported checkpoint formats before trying a shared SUMO run."""

import argparse
import json
from pathlib import Path

import torch

from experiment_env import CONFLICT_OBSERVATIONS, OBS_NAMES


def describe(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    names = tuple(data.get("observation", ()))
    if names == OBS_NAMES and "model" in data:
        mode = data.get("conflict_observation", CONFLICT_OBSERVATIONS[0])
        weights = data["model"].get("lane_head.weight")
        size = data["model"]["trunk.2.bias"].numel()
        return {"format": "role3", "observation_dimension": len(names), "conflict_observation": mode,
                "lane_output": "three_logits", "compatible_with_published_evaluator":
                    mode == CONFLICT_OBSERVATIONS[0] and weights is not None and tuple(weights.shape) == (3, 2 * size)}
    if data.get("algorithm") == "bc" and "state_dict" in data:
        return {"format": "team_legacy_bc", "observation_dimension": data["state_dim"],
                "action_dimension": data["action_dim"], "lane_output": "continuous_scalar",
                "compatible_with_published_evaluator": False}
    return {"format": "unrecognized", "observation_dimension": len(names) or None,
            "compatible_with_published_evaluator": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([{"file": path.name, **describe(path)} for path in args.models], indent=2))


if __name__ == "__main__":
    main()
