"""Collect independently seeded, interruption-safe expert episodes."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EVENTS = (
    "arrived", "collided", "stuck", "off_route", "ego_teleported", "unexpected_exit",
    "red_light", "crosswalk_violation", "yield_violation", "speeding",
    "emergency_seen", "emergency_blocking", "emergency_closing", "front_interaction",
    "junction_conflict", "crossing_occupied", "signal_stop", "stopped_vehicle",
    "background_teleports", "background_collision_vehicles",
)
TRAIN_WINDOWS = (300, 500, 800, 1150, 1330, 1430, 1490)


def json_save(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def prepare_run(tag):
    from experiment_env import OBS_NAMES
    from env.daehakro_builder import build
    run = HERE / "runs" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + tag)
    run.mkdir(parents=True)
    cfg = Path(build(out_dir=str(run / "network"), verbose=False))
    json_save(run / "environment.json", {
        "created_at": datetime.now().astimezone().isoformat(),
        "base_course_commit": "923430f972e4987efc236955e6d5c8af9e1c587f",
        "team_commit": "59c0ae9f8efa3442ff1741b0f518e69b13eb874d",
        "road_and_collector_source": "6a46c548438dacd61555eec5af024c2703e2cdb9",
        "observation": OBS_NAMES,
        "action": ["normalized_acceleration", "left(+1)/keep(0)/right(-1)"],
        "network_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(cfg.parent.iterdir()) if p.is_file()},
        "source_hashes": {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(HERE.rglob("*.py")) if "runs" not in p.parts},
    })
    return run


def dataset_job(run, episode, settings):
    from experiment_env import ExperimentEnv, traci
    folder = run / "expert"
    path = folder / f"{episode:05d}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return json.loads(str(data["summary"]))
    seed = settings["first_seed"] + episode
    env = ExperimentEnv(run / "network/daehakro.sumocfg", teacher=True, seed=seed,
                        start_time=TRAIN_WINDOWS[episode % len(TRAIN_WINDOWS)],
                        native_teacher=settings.get("native_teacher", False))
    env.demo_noise = settings["noise"]
    observations, actions, timesteps = [], [], []
    counts = dict.fromkeys(EVENTS, 0)
    try:
        obs, _ = env.reset()
        started = traci.simulation.getTime()
        counts["unexpected_exit"] = 0
        background = []
        for t in range(env.max_steps):
            next_obs, _, terminated, truncated, info = env.step([0., 0.])
            assert np.array_equal(info["demo_obs"], obs)
            action = info["demo_action"]
            if action is not None:
                assert action[1] == np.sign(info["lane_changed"])
                observations.append(obs.copy())
                actions.append(action)
                timesteps.append(t)
            for key in EVENTS:
                counts[key] += int(info[key])
            background.append(info["background_vehicles"])
            obs = next_obs
            if terminated or truncated:
                break
        summary = {
            "episode": episode, "seed": seed, "start_time": started, "steps": t + 1,
            "rows": len(actions), "background_mean": float(np.mean(background)),
            "look_errors": env.look_errors, "events": counts,
        }
        with path.with_suffix(".tmp").open("wb") as file:
            np.savez_compressed(file, observation=np.asarray(observations, dtype=np.float32),
                                action_raw=np.asarray(actions, dtype=np.float32),
                                t=np.asarray(timesteps, dtype=np.int32),
                                summary=json.dumps(summary))
        os.replace(path.with_suffix(".tmp"), path)
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Continue an existing collection")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=110000)
    parser.add_argument("--noise", type=float, default=.005)
    parser.add_argument("--native-teacher", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve() if args.run else prepare_run("bc_ppo")
    config = run / "collection.json"
    settings = {"first_seed": args.seed, "episodes": args.episodes, "noise": args.noise,
                "native_teacher": args.native_teacher}
    if config.exists():
        settings = json.loads(config.read_text(encoding="utf-8"))
    else:
        json_save(config, settings)
    (run / "expert").mkdir(exist_ok=True)
    print(f"RUN {run}", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run_dataset_job, run, ep, settings) for ep in range(settings["episodes"])]
        for future in concurrent.futures.as_completed(pending):
            print(json.dumps(future.result()), flush=True)


def run_dataset_job(run, episode, settings):
    from experiment_env import InsertionError
    error_path = run / "expert" / f"{episode:05d}.error.json"
    if error_path.exists():
        return json.loads(error_path.read_text(encoding="utf-8"))
    try:
        return dataset_job(run, episode, settings)
    except InsertionError as exc:
        failure = {"episode": episode, "seed": settings["first_seed"] + episode,
                   "status": "insertion_failed", "reason": str(exc)}
        json_save(error_path, failure)
        return failure
    except Exception as exc:
        raise RuntimeError(f"Expert episode {episode}: {exc}") from None


if __name__ == "__main__":
    main()
