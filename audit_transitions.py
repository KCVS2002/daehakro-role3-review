"""Check that exported transitions match the course schema and their recorded timeline."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from collect_expert import json_save
from env.mdp_config import OBSERVATION
from experiment_env import OBS_NAMES, signed_lane_turns


REQUIRED = {
    "episode", "t", "observation", "action_raw", "action", "reward", "terminated",
    "truncated", "state", "detected_vehicles", "event", "next_observation", "next_state",
}


def audit(path, signed_turn, actor=None):
    from evaluate_policy import predict

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    for row, successor in zip(rows, rows[1:]):
        row.setdefault("next_observation", successor["observation"])
        row.setdefault("next_state", successor["state"])
    with np.load(path.with_name(path.name.removesuffix(".jsonl.gz") + ".npz"), allow_pickle=False) as data:
        assert len(rows) == len(data["observation"])
        assert data["detected_offsets"][0] == 0
        assert data["detected_offsets"][-1] == len(data["detected_xy"])
        assert np.array_equal(np.diff(data["detected_offsets"]), [len(r["detected_vehicles"]) for r in rows])
        assert np.allclose(data["observation"][1:], data["next_observation"][:-1], atol=0, rtol=0)
        assert data["action_raw"].shape == (len(rows), 2)
        assert np.all(np.isfinite(data["observation"]))
        for key in ("episode", "t", "observation", "next_observation", "action_raw", "reward",
                    "terminated", "truncated"):
            np.testing.assert_array_equal(data[key], np.asarray([row[key] for row in rows], dtype=data[key].dtype))
        np.testing.assert_array_equal(data["lane_change"], [row["action"]["lane_change"] for row in rows])
        expected_xy = [(v["x"], v["y"]) for row in rows for v in row["detected_vehicles"]]
        np.testing.assert_array_equal(data["detected_xy"], np.asarray(expected_xy, dtype=np.float32).reshape(-1, 2))
    largest_acceleration_difference = 0.
    for i, row in enumerate(rows):
        assert REQUIRED <= row.keys()
        assert i == row["t"] and row["episode"] == rows[0]["episode"]
        assert {"accel_raw", "lane_change_raw", "lane_change", "lane_change_applied"} <= row["action"].keys()
        assert {"collided", "arrived", "lane_before", "lane_after"} <= row["event"].keys()
        assert [row["action"]["accel_raw"], row["action"]["lane_change_raw"]] == row["action_raw"]
        assert row["action"]["lane_change"] in (-1, 0, 1)
        if actor is not None:
            acceleration, lane = predict(np.asarray(row["observation"], dtype=np.float32), actor)
            error = abs(float(acceleration) - row["action_raw"][0])
            if error > 4.e-6 or lane != row["action_raw"][1]:
                raise ValueError(f"Recorded action differs from the exported actor at {path.name}:{i}")
            largest_acceleration_difference = max(largest_acceleration_difference, error)
        assert np.isfinite(row["reward"])
        assert not (row["event"]["collided"] and row["event"]["arrived"])
        assert isinstance(row["state"]["vehicles"], list) and isinstance(row["state"]["signals"], dict)
        assert isinstance(row["state"]["pedestrians"], list)
        state = row["state"]
        assert state["ego_alive"]
        ego_id = next(v["id"] for v in state["vehicles"]
                      if (v["lane_id"], v["x"], v["y"]) == (state["ego_lane_id"], state["ego_x"], state["ego_y"]))
        assert len({v["id"] for v in state["vehicles"]}) == len(state["vehicles"])
        expected_vehicles = [v for v in state["vehicles"] if v["id"] != ego_id
                             and (v["x"] - state["ego_x"]) ** 2 + (v["y"] - state["ego_y"]) ** 2
                             <= OBSERVATION["visibility"] ** 2]
        assert row["detected_vehicles"] == expected_vehicles
        assert abs(row["next_state"]["sim_time_s"] - row["state"]["sim_time_s"] - .5) < 1e-6
        for state, obs in ((row["state"], row["observation"]), (row["next_state"], row["next_observation"])):
            assert len(obs) == len(OBS_NAMES)
            expected = signed_turn.get(state.get("ego_lane_id"), 0.)
            assert abs(obs[-1] - expected) <= 1e-7
        if i + 1 < len(rows):
            assert not (row["terminated"] or row["truncated"])
            assert row["next_state"] == rows[i + 1]["state"]
            assert row["next_observation"] == rows[i + 1]["observation"]
    assert rows[-1]["terminated"] or rows[-1]["truncated"]
    report = {"file": path.name, "transitions": len(rows), "observation_dim": len(rows[0]["observation"]),
              "curved_lane_observations": sum(row["observation"][-1] != 0. for row in rows),
              "matched_json_npz_rows": len(rows), "checked_detected_vehicle_rows": len(rows),
              "initial_background_vehicles": len(rows[0]["state"]["vehicles"]) - 1,
              "signal_controllers": len(rows[0]["state"]["signals"]), "final_event": rows[-1]["event"]}
    if actor is not None:
        report.update(checked_policy_rows=len(rows), largest_acceleration_difference=largest_acceleration_difference)
    return report


def verified_actor(folder, config):
    from evaluate_policy import export_actor

    checkpoint = Path(config["model"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != config["model_sha256"]:
        raise ValueError("Model fingerprint differs from the evaluated checkpoint")
    with TemporaryDirectory() as tmp:
        expected = Path(tmp) / "actor.npz"
        export_actor(checkpoint, expected, config.get("conflict_observation"))
        with np.load(expected, allow_pickle=False) as new, np.load(folder / "actor.npz", allow_pickle=False) as old:
            if set(new.files) != set(old.files):
                raise ValueError("Actor snapshot has a different set of weights")
            for key in new.files:
                np.testing.assert_array_equal(new[key], old[key])
            return dict(old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--verify-policy", action="store_true", help="Check recorded actions against the evaluated checkpoint")
    args = parser.parse_args()
    config = json.loads((args.folder / "config.json").read_text(encoding="utf-8"))
    turns = signed_lane_turns(Path(config["network"]).with_name("daehakro.net.xml"))
    actor = verified_actor(args.folder, config) if args.verify_policy else None
    results = [audit(path, turns, actor) for path in sorted(args.folder.glob("*.jsonl.gz"))]
    if not results:
        raise ValueError("No transitions to check")
    json_save(args.folder / "transition_audit.json", results)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
