"""Headless held-out evaluation with optional course-format transition samples."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from collect_expert import EVENTS, json_save
from experiment_env import CONFLICT_OBSERVATIONS, TELEPORT_TIMEOUT_S, sumo_input_hashes


WINDOWS = (300, 650, 1000, 1360, 1450)


def export_actor(checkpoint, target, conflict_observation=None):
    import torch
    from experiment_env import OBS_NAMES
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if tuple(data.get("observation", ())) != OBS_NAMES:
        raise ValueError("Checkpoint does not declare the ordered 65-field Role 3 observation schema")
    mode = data.get("conflict_observation")
    if conflict_observation is not None and mode is not None and mode != conflict_observation:
        raise ValueError("The saved actor uses a different conflict observation")
    required = ("obs_mean", "obs_scale", "trunk.0.weight", "trunk.0.bias",
                "trunk.2.weight", "trunk.2.bias", "mu_head.weight", "mu_head.bias",
                "lane_head.weight", "lane_head.bias")
    snapshot = {key: data["model"][key].numpy() for key in required}
    size = len(snapshot["trunk.2.bias"])
    lane_weights = snapshot["lane_head.weight"]
    if lane_weights.shape != (3, 2 * size):
        raise ValueError("The lane checkpoint lacks the learned geometry interactions")
    snapshot["lane_head.weight"] = np.ascontiguousarray(lane_weights[:, :size])
    snapshot["curvature_lane_weights"] = np.ascontiguousarray(lane_weights[:, size:])
    if "crossing_accel.weight" in data["model"]:
        local = data["model"]["crossing_accel.weight"].numpy()
        if local.shape != (size + 1,):
            raise ValueError("The crossing acceleration weights have an invalid shape")
        snapshot["crossing_accel.weight"] = local
    np.savez(target, **snapshot)


def crossing_proximity(distance):
    return 1. - distance * distance


def predict(obs, actor):
    from experiment_env import PREVIOUS_OBS_NAMES

    dimension = len(PREVIOUS_OBS_NAMES)
    mean, scale = actor["obs_mean"], actor["obs_scale"]
    # A 64-input actor is also supported for exact, offline teacher replays.
    x = np.clip((obs[:len(mean)] - mean) / scale, -10, 10)
    curvature = x[dimension] if len(x) > dimension else 0.
    x = x[:dimension]
    for layer in ("trunk.0", "trunk.2"):
        x = np.maximum(actor[layer + ".weight"] @ x + actor[layer + ".bias"], 0)
    acceleration = actor["mu_head.weight"] @ x + actor["mu_head.bias"]
    if "crossing_accel.weight" in actor:
        distance = obs[PREVIOUS_OBS_NAMES.index("crossing_distance_until_rear_clear")]
        local = actor["crossing_accel.weight"]
        acceleration += np.float32(crossing_proximity(distance) * (local[:-1] @ x + local[-1]))
    mu = np.tanh(acceleration)[0]
    logits = actor["lane_head.weight"] @ x + actor["lane_head.bias"]
    if "curvature_lane_weights" in actor:
        logits = logits + actor["curvature_lane_weights"] @ (curvature * x)
    lane = 1 - np.argmax(logits)
    return np.array((mu, lane), dtype=np.float32)


def detected(env, state):
    ex, ey = state["ego_x"], state["ego_y"]
    return [veh for veh in state["vehicles"] if veh["id"] != env.ego_id
            and (veh["x"] - ex) ** 2 + (veh["y"] - ey) ** 2 <= env.visibility ** 2]


def save_transitions(out, episode, rows):
    path = out / f"{episode:04d}.jsonl.gz"
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=4) as file:
        for i, row in enumerate(rows):
            stored = row if i == len(rows) - 1 else {
                key: val for key, val in row.items() if key not in ("next_state", "next_observation")
            }
            file.write(json.dumps(stored, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(tmp, path)
    path = out / f"{episode:04d}.npz"
    offsets = np.cumsum([0] + [len(row["detected_vehicles"]) for row in rows])
    coords = [(d["x"], d["y"]) for row in rows for d in row["detected_vehicles"]]
    data = {key: np.asarray([row[key] for row in rows]) for key in (
        "observation", "action_raw", "reward", "next_observation", "terminated", "truncated",
        "episode", "t",
    )}
    with path.with_suffix(".tmp").open("wb") as file:
        np.savez_compressed(file, **data,
                            lane_change=[row["action"]["lane_change"] for row in rows],
                            detected_xy=np.asarray(coords, dtype=np.float32).reshape(-1, 2),
                            detected_offsets=offsets)
    os.replace(path.with_suffix(".tmp"), path)


def evaluate_job(network, out, episode, seed, time, dump, collision_gap_factor,
                 conflict_observation=CONFLICT_OBSERVATIONS[0]):
    from experiment_env import ExperimentEnv, InsertionError
    from env.daehakro_env import traci
    path = out / f"{episode:04d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with np.load(out / "actor.npz", allow_pickle=False) as arrays:
        actor = dict(arrays)
    env = ExperimentEnv(network, seed=seed, start_time=time, collision_gap_factor=collision_gap_factor,
                        conflict_observation=conflict_observation)
    counts = dict.fromkeys(EVENTS, 0)
    rows, background, edges, interaction_edges = [], [], set(), {}
    low_speed_streak = max_low_speed_streak = 0
    total_reward = 0.0
    lane_changes = 0
    interaction_responses = {}
    collision = None
    try:
        obs, _ = env.reset()
        departure = traci.simulation.getTime()
        state = env.get_privileged_state() if dump else None
        for t in range(env.max_steps):
            seen = detected(env, state) if dump else None
            edge = traci.vehicle.getRoadID(env.ego_id)
            speed = traci.vehicle.getSpeed(env.ego_id)
            edges.add(edge)
            action = predict(obs, actor)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            lane_changes += int(info["lane_changed"] != 0)
            low_speed_streak = low_speed_streak + 1 if speed < .1 else 0
            max_low_speed_streak = max(max_low_speed_streak, low_speed_streak)
            for key in EVENTS:
                counts[key] += int(info[key])
                if info[key] and key in ("front_interaction", "junction_conflict",
                                         "crossing_occupied", "signal_stop", "stopped_vehicle",
                                         "emergency_seen", "emergency_blocking"):
                    interaction_edges.setdefault(key, set()).add(edge)
                    response = interaction_responses.setdefault(key, dict(
                        braking_steps=0, stopped_steps=0, lane_changes=0))
                    response["stopped_steps"] += int(speed < .1)
                    response["braking_steps"] += int(info["speed"] is not None and info["speed"] < speed - .2)
                    response["lane_changes"] += int(info["lane_changed"] != 0)
            if info["collided"]:
                collision = {"edge": edge, "with": info.get("collision_with"),
                             "type": info.get("collision_type"), "speed_mps": speed}
            background.append(info["background_vehicles"])
            if dump:
                next_state = env.get_privileged_state()
                rows.append({
                    "episode": episode, "t": t, "observation": obs.tolist(), "action_raw": action.tolist(),
                    "action": {key: info[key] for key in
                               ("accel_raw", "lane_change_raw", "lane_change", "lane_change_applied")},
                    "reward": reward, "terminated": bool(terminated), "truncated": bool(truncated),
                    "state": state, "detected_vehicles": seen,
                    "event": {key: info[key] for key in
                              ("collided", "arrived", "lane_before", "lane_after", "off_route",
                               "red_light", "crosswalk_violation", "yield_violation",
                               "emergency_seen", "emergency_blocking", "ego_teleported")} | {
                                   key: info[key] for key in (
                                       "signal_phase_before_crossing", "signal_phase_during_crossing")
                                   if key in info
                               },
                    "next_observation": next_obs.tolist(), "next_state": next_state,
                })
                state = next_state
            obs = next_obs
            if terminated or truncated:
                break
        if dump:
            save_transitions(out, episode, rows)
        result = {
            "episode": episode, "seed": seed, "requested_start_s": time,
            "depart_s": departure, "duration_s": (t + 1) * env.step_length,
            "reward": total_reward, "lane_changes": lane_changes,
            "max_stop_s": max_low_speed_streak * env.step_length,
            "events": counts, "last_edge": edge, "edges": sorted(edges),
            "interaction_edges": {k: sorted(v) for k, v in interaction_edges.items()},
            "interaction_responses": interaction_responses, "collision": collision,
            "background_mean": float(np.mean(background)), "background_min": min(background),
            "look_errors": env.look_errors, "terminated": bool(terminated), "truncated": bool(truncated),
        }
        json_save(path, result)
        return result
    except InsertionError as exc:
        result = {
            "episode": episode, "seed": seed, "requested_start_s": time,
            "depart_s": None, "duration_s": 0.0, "reward": 0.0, "lane_changes": 0,
            "max_stop_s": 0.0, "events": counts, "last_edge": None, "edges": [],
            "interaction_edges": {}, "interaction_responses": {}, "collision": None,
            "background_mean": float(len(traci.vehicle.getIDList())), "background_min": None,
            "look_errors": env.look_errors, "terminated": False, "truncated": True,
            "insertion_failed": str(exc),
        }
        json_save(path, result)
        return result
    finally:
        env.close()


def summarize(results):
    arrived = [r for r in results if r["events"]["arrived"] and not r["events"]["collided"]]
    return {
        "episodes": len(results),
        "insertion_failures": sum(bool(r.get("insertion_failed")) for r in results),
        "episodes_with_events": {key: sum(r["events"][key] > 0 for r in results) for key in EVENTS},
        "total_events_or_steps": {key: sum(r["events"][key] for r in results) for key in EVENTS},
        "collision_rate": sum(r["events"]["collided"] > 0 for r in results) / len(results),
        "arrival_rate": len(arrived) / len(results),
        "clean_arrivals": sum(not any(r["events"][key] for key in (
            "collided", "red_light", "crosswalk_violation", "yield_violation", "speeding",
            "ego_teleported", "unexpected_exit")) for r in arrived),
        "arrival_mean_s": float(np.mean([r["duration_s"] for r in arrived])) if arrived else None,
        "background_mean": float(np.mean([r["background_mean"] for r in results])),
        "lane_changes": sum(r["lane_changes"] for r in results),
        "longest_stop_s": max(r["max_stop_s"] for r in results),
        "look_errors": sum(r["look_errors"] for r in results),
    }


def run_evaluation(network, out, episode, seed, time, dump, collision_gap_factor,
                   conflict_observation=CONFLICT_OBSERVATIONS[0]):
    try:
        return evaluate_job(network, out, episode, seed, time, dump, collision_gap_factor, conflict_observation)
    except Exception as exc:
        raise RuntimeError(f"Evaluation episode {episode}, seed {seed}: {exc}") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=310000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--windows", type=float, nargs="+", default=WINDOWS)
    parser.add_argument("--dump", type=int, default=0, help="Record full transitions for the first N episodes")
    parser.add_argument("--collision-gap-factor", type=float, default=-1.0,
                        help="0 records physical contact, -1 uses the SUMO vehicle-type defaults")
    parser.add_argument("--conflict-observation", choices=CONFLICT_OBSERVATIONS, default=CONFLICT_OBSERVATIONS[0])
    parser.add_argument("--resume", type=Path, help="Continue an interrupted evaluation directory")
    args = parser.parse_args()
    network = args.run.resolve() / "network/daehakro.sumocfg"
    network_hashes = sumo_input_hashes(network)
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if args.resume:
        out = args.resume.resolve()
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        if (config["model_sha256"] != model_hash or config["network"] != str(network)
                or config.get("network_sha256") != network_hashes
                or config.get("teleport_timeout_s") != TELEPORT_TIMEOUT_S
                or config.get("conflict_observation", CONFLICT_OBSERVATIONS[0]) != args.conflict_observation):
            raise ValueError("The model, network, observation or teleport setting differs from the saved evaluation")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = Path(tempfile.mkdtemp(
            prefix=f"{timestamp}_eval_{args.model.stem}_", dir=args.model.resolve().parent))
        export_actor(args.model, out / "actor.npz", args.conflict_observation)
        config = {
            "network": str(network), "model": str(args.model.resolve()), "model_sha256": model_hash,
            "network_sha256": network_hashes,
            "collision_gap_factor": args.collision_gap_factor,
            "teleport_timeout_s": TELEPORT_TIMEOUT_S,
            "conflict_observation": args.conflict_observation,
            "cases": [{"episode": ep, "seed": args.seed + ep,
                       "start_time": args.windows[ep % len(args.windows)], "dump": ep < args.dump}
                      for ep in range(args.episodes)],
        }
        json_save(out / "config.json", config)
    jobs = [(out, case["episode"], case["seed"], case["start_time"], case["dump"])
            for case in config["cases"]]
    print(f"EVAL {out}", flush=True)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_evaluation, network, *job, config.get("collision_gap_factor", 0.0),
                               config.get("conflict_observation", CONFLICT_OBSERVATIONS[0]))
                   for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    summary = {"overall": summarize(results),
               "by_start_time": {str(w): summarize([r for r in results if r["requested_start_s"] == w])
                                 for w in sorted({r["requested_start_s"] for r in results})}}
    json_save(out / "summary.json", summary)
    print(json.dumps(summary["overall"]), flush=True)


if __name__ == "__main__":
    main()
