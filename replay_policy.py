"""Replay the verified AV showcase under the published SUMO conditions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="Open SUMO GUI and follow the AV")
    parser.add_argument("--delay", type=int, default=85, help="Display delay per simulated step in milliseconds")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("Display delay must be nonnegative")
    if args.gui:
        os.environ["DAEHAKRO_USE_TRACI"] = "1"
        os.environ.pop("LIBSUMO_AS_TRACI", None)

    from evaluate_policy import EVENTS, predict
    from experiment_env import ExperimentEnv, sumo_input_hashes, traci

    case = json.loads((ROOT / "evidence/demo_case.json").read_text(encoding="utf-8"))
    network = ROOT / case["network"]
    actor_file = ROOT / case["actor"]
    if sumo_input_hashes(network) != case["network_sha256"]:
        raise ValueError("Showcase road inputs differ from the evaluated case")
    if hashlib.sha256(actor_file.read_bytes()).hexdigest() != case["actor_sha256"]:
        raise ValueError("Showcase actor differs from the evaluated case")
    with np.load(actor_file, allow_pickle=False) as archive:
        actor = dict(archive)

    env = ExperimentEnv(network, seed=case["seed"], start_time=case["requested_start_s"], gui=args.gui,
                        collision_gap_factor=case["collision_gap_factor"],
                        conflict_observation=case["conflict_observation"])
    events = dict.fromkeys(EVENTS, 0)
    try:
        print(f"Preparing traffic at t={case['requested_start_s']} s (seed {case['seed']})", flush=True)
        obs, _ = env.reset()
        departure_s = traci.simulation.getTime()
        if args.gui:
            view = traci.gui.getIDList()[0]
            traci.gui.trackVehicle(view, env.ego_id)
        lane_changes = 0
        for steps in range(1, env.max_steps + 1):
            obs, _, terminated, truncated, info = env.step(predict(obs, actor))
            for event in events:
                events[event] += int(info[event])
            lane_changes += int(info["lane_changed"] != 0)
            if terminated or truncated:
                break
            if args.gui:
                time.sleep(args.delay / 1000.0)
        result = {"depart_s": departure_s, "duration_s": steps * env.step_length,
                  "lane_changes": lane_changes, "events": events}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if any(case["expected"][key] != result[key] for key in ("depart_s", "duration_s", "lane_changes")) or any(
                case["expected"]["events"][key] != count for key, count in events.items()):
            raise RuntimeError("Replay outcome differs from the archived evaluation")
        if args.gui:
            input("Replay complete. Press Enter to close the viewer.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
