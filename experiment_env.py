"""Local role 3 environment based on the team's 6a46c54 road."""
import os
if os.environ.get("DAEHAKRO_USE_TRACI") != "1":
    os.environ.setdefault("LIBSUMO_AS_TRACI", "quiet")

import hashlib
import math
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
from gymnasium import spaces

from env import daehakro_env as upstream
from env.daehakro_env import DaehakroEnv, RED_STATES, tc, traci
from env.mdp_config import ACTION, OBSERVATION, REWARD, SIMULATION
from sumolib import checkBinary


EXTRAS = (
    "speed_limit", "signal_distance", "signal_red", "signal_yellow",
    "crossing_distance_until_rear_clear", "crossing_occupied_until_rear_clear", "junction_distance",
    "conflict_vehicle_distance", "conflict_vehicle_speed", "conflict_vehicle_present",
    "emergency_distance", "emergency_behind_same_lane", "emergency_relative_speed",
    "lane_index", "route_distance_fraction", "segment_remaining",
    "left_lane_marking", "right_lane_marking",
)
CONTEXT = (
    "nearest_person_present", "nearest_person_forward", "nearest_person_right",
    "coarse_route_waypoint_forward", "coarse_route_waypoint_right",
)
OFFSETS = (2, 1, 0, -1, -2)
CURRENT_OBS_NAMES = (
    ("ego_speed_over_vehicle_maximum",)
    + tuple(f"lane_{o}_{field}" for o in OFFSETS
            for field in ("front_gap", "front_relative_speed", "rear_gap", "rear_relative_speed"))
    + tuple(f"lane_{o}_exists" for o in OFFSETS)
    + EXTRAS + CONTEXT
)
HISTORY_FIELDS = (
    "ego_speed_over_vehicle_maximum", "lane_0_front_gap", "lane_-1_front_gap",
    "lane_1_front_gap", "signal_distance", "signal_red",
    "crossing_distance_until_rear_clear", "crossing_occupied_until_rear_clear",
    "nearest_person_present", "nearest_person_forward", "nearest_person_right",
    "conflict_vehicle_distance", "emergency_distance", "emergency_behind_same_lane",
    "emergency_relative_speed",
)
HISTORY_INDICES = [CURRENT_OBS_NAMES.index(name) for name in HISTORY_FIELDS]
PREVIOUS_OBS_NAMES = CURRENT_OBS_NAMES + tuple(f"delta_1s_{name}" for name in HISTORY_FIELDS)
OBS_NAMES = PREVIOUS_OBS_NAMES + ("signed_current_lane_turn_per_10m",)
TELEPORT_TIMEOUT_S = -1
CONFLICT_OBSERVATIONS = ("yield-filtered", "all-approaches")


def sumo_input_hashes(config):
    config = Path(config).resolve()
    network = ET.parse(config).find("input")
    if network is None:
        raise ValueError(f"Missing SUMO inputs in {config}")
    paths = {config.name: config}
    for field in ("net-file", "route-files", "additional-files"):
        element = network.find(field)
        if element is not None:
            for name in element.attrib["value"].split(","):
                paths[name.strip()] = config.parent / name.strip()
    return {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sorted(paths.items())}


def signed_lane_turns(net_file):
    result = {}
    for lane in ET.parse(net_file).getroot().iter("lane"):
        shape = [tuple(map(float, item.split(",")[:2])) for item in lane.get("shape", "").split()]
        segments = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(shape, shape[1:]) if a != b]
        length = sum(math.hypot(x, y) for x, y in segments)
        turns = (math.atan2(x * v - y * u, x * u + y * v) for (x, y), (u, v) in zip(segments, segments[1:]))
        rate = math.degrees(sum(turns)) * 10. / length if length else 0.
        result[lane.get("id")] = max(-1., min(1., rate / 20.))
    return result


def road_centers(net_file, passenger_lanes):
    """A coarse reference point per major road piece, with no lane identity."""
    centers = {}
    for edge in ET.parse(net_file).getroot().iter("edge"):
        if edge.get("function") in ("internal", "crossing", "walkingarea"):
            continue
        endpoints = []
        for lane in edge.iter("lane"):
            if not passenger_lanes.get(lane.get("id"), False):
                continue
            shape = lane.get("shape", "").split()
            if shape:
                endpoints.extend(tuple(map(float, point.split(",")[:2])) for point in (shape[0], shape[-1]))
        if endpoints:
            x, y = np.mean(endpoints, axis=0)
            centers[edge.get("id")] = (round(x / 10) * 10., round(y / 10) * 10.)
    return centers


class InsertionError(RuntimeError):
    """Traffic blocked the route entry throughout the insertion window."""


class ExperimentEnv(DaehakroEnv):
    def __init__(self, cfg_path, teacher=False, seed=0, start_time=300.0,
                 collision_gap_factor=-1.0, native_teacher=False, gui=False,
                 conflict_observation=CONFLICT_OBSERVATIONS[0]):
        if conflict_observation not in CONFLICT_OBSERVATIONS:
            raise ValueError(f"Unknown conflict observation: {conflict_observation}")
        self.conflict_observation = conflict_observation
        super().__init__(str(cfg_path), None, None, SIMULATION, OBSERVATION,
                         ACTION, REWARD, gui=gui, gui_autostart=gui, demo=teacher)
        assert tuple(self.obs_offsets) == OFFSETS
        self.obs_dim = len(OBS_NAMES)
        self.observation_space = spaces.Box(-1.0, 1.0, (self.obs_dim,), dtype=np.float32)
        self._observation_history = deque(maxlen=3)
        self.lc_safety_enabled = teacher
        self.physical_max_speed = 22.0
        self.sim_seed = seed
        self.start_time = start_time
        self.collision_gap_factor = collision_gap_factor
        self._rng = np.random.default_rng(seed)
        routes = ET.parse(Path(cfg_path).with_name("daehakro.rou.xml"))
        self.emergency_ids = tuple(v.attrib["id"] for v in routes.iter("vehicle")
                                   if v.get("type", "").endswith("_siren"))
        self.assist = None
        self.emergency = None
        self.native_teacher = teacher and native_teacher
        self.road_centers = road_centers(Path(cfg_path).with_name("daehakro.net.xml"), self.map.car_ok)
        self.lane_turns = signed_lane_turns(Path(cfg_path).with_name("daehakro.net.xml"))

    def reset(self, seed=None, options=None):
        self._observation_history.clear()
        obs, info = super().reset(seed=seed, options=options)
        if self.native_teacher:
            traci.vehicle.setSpeedMode(self.ego_id, 31)
            traci.vehicle.setLaneChangeMode(self.ego_id, 1621)
            traci.vehicle.setParameter(self.ego_id, "laneChangeModel.lcStrategic", "1")
        return obs, info

    def _demo_command(self, state):
        if self.native_teacher:
            return [1.0, 0.0], False
        return super()._demo_command(state)

    def _sumo_command(self):
        return [
            checkBinary("sumo-gui" if self.gui else "sumo"),
            "-c", self.cfg_path, "--seed", str(self.sim_seed),
            "--no-step-log", "true", "--no-warnings", "true",
            "--time-to-teleport", str(TELEPORT_TIMEOUT_S),
            "--collision.action", "remove", "--collision.check-junctions", "true",
            "--collision.mingap-factor", str(self.collision_gap_factor),
        ]

    def _start_sumo(self):
        command = self._sumo_command()
        if self.gui:
            command += ["--start", "--delay", "0"]
        traci.start(command)
        self._started = True

    def _restart(self):
        if self._started:
            traci.close()
            self._started = False
        self._start_sumo()
        while traci.simulation.getTime() < self.start_time:
            self._advance()

    def _advance(self):
        traci.simulationStep()

    def _insert_ego(self):
        if not super()._insert_ego():
            raise InsertionError(f"AV insertion blocked at seed {self.sim_seed}, start {self.start_time}s")
        return True

    def _siren(self):
        self.emergency = None
        ex, ey = traci.vehicle.getPosition(self.ego_id)
        lane = traci.vehicle.getLaneID(self.ego_id)
        speed = traci.vehicle.getSpeed(self.ego_id)
        ego_angle = math.radians(traci.vehicle.getAngle(self.ego_id))
        live = set(traci.vehicle.getIDList())
        nearest = None
        for vid in self.emergency_ids:
            if vid not in live:
                continue
            x, y = traci.vehicle.getPosition(vid)
            distance = math.hypot(x - ex, y - ey)
            if distance > upstream.SIREN_RANGE or (nearest and distance >= nearest[0]):
                continue
            angle = math.radians(traci.vehicle.getAngle(vid))
            behind = (ex - x) * math.sin(angle) + (ey - y) * math.cos(angle) > 0
            same_lane = lane == traci.vehicle.getLaneID(vid)
            nearest = distance, bool(behind and same_lane and math.cos(angle - ego_angle) > .5)
            self.emergency = (traci.vehicle.getSpeed(vid) - speed) / self.vmax
        return nearest

    def _lane_connectivity(self, offset):
        lane = self._offset_lane(offset)
        return float(self.map.car_ok.get(lane, False)) if lane is not None else 0.0

    def _visible_conflict(self):
        nearest = None
        for vehicle, ego_dist, dist, ego_out, foe_out, *_ in traci.vehicle.getJunctionFoes(
                self.ego_id, self.visibility):
            if ego_out <= 0.0 or foe_out <= 0.0 or max(ego_dist, dist) > self.visibility:
                continue
            speed = traci.vehicle.getSpeed(vehicle)
            key = max(dist, 0.) / max(speed, .1), max(ego_dist, 0.), vehicle
            if nearest is None or key < nearest[0]:
                nearest = key, (ego_dist, max(dist, 0.), speed)
        return nearest[1] if nearest is not None else None

    def _extras(self, s):
        tls, cw, foe, siren = s["tls"], s["cw"], s["foe"], s["siren"]
        if foe is None and self.conflict_observation == "all-approaches":
            foe = self._visible_conflict()
        left, right = self.map.change_ok.get(s["lane"], (False, False))
        return [
            s["limit"] / self.vmax,
            upstream._near(tls[1], upstream.SIGNAL_RANGE) if tls else 1.0,
            float(bool(tls and tls[2] in RED_STATES)),
            float(bool(tls and tls[2] in "yY")),
            upstream._near(cw[1], self.visibility) if cw else 1.0,
            float(bool(cw and cw[2])),
            upstream._near(foe[0], self.visibility) if foe else 1.0,
            foe[1] / self.visibility if foe else 1.0,
            foe[2] / self.vmax if foe else 0.0,
            float(foe is not None),
            siren[0] / upstream.SIREN_RANGE if siren else 1.0,
            float(bool(siren and siren[1])),
            self.emergency if siren else 0.0,
            traci.vehicle.getLaneIndex(self.ego_id) / 5.0,
            s["odo"] / 1516.0,
            upstream._near(self.map.length.get(s["lane"], 0.0) - s["pos"], self.visibility),
            float(left),
            float(right),
        ]

    def _get_obs(self):
        if self.ego_id not in traci.vehicle.getIDList():
            self._snap = None
            self._last_obs = np.zeros(self.obs_dim, dtype=np.float32)
        else:
            self._snap = self._look()
            base = super(DaehakroEnv, self)._get_obs()[:26]
            base[0] = self._snap["v"] / self.physical_max_speed
            # Preserve resolution near neighboring bumpers.
            gap_indices = np.arange(1, 21, 2)
            base[gap_indices] = np.sign(base[gap_indices]) * np.sqrt(np.abs(base[gap_indices]))
            extras = np.asarray(self._extras(self._snap), dtype=np.float32)
            context = np.asarray(self._local_context(), dtype=np.float32)
            current = np.clip(np.concatenate([base, extras, context]), -1.0, 1.0)
            curvature = np.float32(self.lane_turns.get(self._snap["lane"], 0.))
            self._last_obs = np.concatenate([current, self._history_delta(current), [curvature]])
        return self._last_obs

    def _history_delta(self, observation):
        now = traci.simulation.getTime()
        current = observation[HISTORY_INDICES]
        history = self._observation_history
        if history and history[-1][0] == now:
            history[-1] = (now, current)
        else:
            history.append((now, current))
        previous_time, previous = history[0]
        if not math.isclose(now - previous_time, 1.0, abs_tol=1e-6):
            return np.zeros_like(current)
        return (current - previous) * .5

    def _local_context(self):
        ego = self.ego_id
        ex, ey = traci.vehicle.getPosition(ego)
        angle = math.radians(traci.vehicle.getAngle(ego))
        sine, cosine = math.sin(angle), math.cos(angle)

        def frame(x, y):
            dx, dy = x - ex, y - ey
            return dx * sine + dy * cosine, dx * cosine - dy * sine

        nearest, best = None, 20.0
        for data in (traci.vehicle.getContextSubscriptionResults(ego) or {}).values():
            if tc.VAR_NEXT_EDGE not in data or tc.VAR_POSITION not in data:
                continue
            forward, right = frame(*data[tc.VAR_POSITION])
            distance = math.hypot(forward, right)
            if distance < best:
                nearest, best = (forward, right), distance
        person = (1., nearest[0] / 20., nearest[1] / 20.) if nearest is not None else (0., 0., 0.)

        waypoint = (0., 0.)
        index = traci.vehicle.getRouteIndex(ego)
        for edge in self._route[max(index + 1, 0):]:
            if edge in self.road_centers:
                waypoint = frame(*self.road_centers[edge])
                if math.hypot(*waypoint) >= 45.0:
                    break
        return (*person, waypoint[0] / 80., waypoint[1] / 80.)

    def _neighbor(self, offset, leader):
        if offset == 0 and leader:
            lead = traci.vehicle.getLeader(self.ego_id, self.visibility)
            if not lead or not lead[0] or lead[1] > self.visibility:
                return None
            return max(lead[1] + self._gap_shift, 0.0), traci.vehicle.getSpeed(lead[0])
        return super()._neighbor(offset, leader)

    def _look(self):
        state = super()._look()
        lead = self._neighbor(0, True)
        if lead is not None:
            state["gap"], state["lead_v"] = lead
        return state

    def _apply_lane_change(self, lane_change):
        if self.demo:
            return super()._apply_lane_change(lane_change)
        if lane_change == 0:
            return True
        lane = self._snap["lane"]
        if lane.startswith(":"):
            return False
        edge, index = lane.rsplit("_", 1)
        target = int(index) + lane_change
        markings = self.map.change_ok.get(lane, (True, True))
        if not self.map.car_ok.get(f"{edge}_{target}", False) or not markings[lane_change < 0]:
            return False
        traci.vehicle.changeLane(self.ego_id, target, self.lane_change_duration)
        return True

    def step(self, action):
        before = self._snap
        obs, reward, terminated, truncated, info = super().step(action)
        # Training can discourage stopping with the body still on the crossing.
        # The original action reaches SUMO before this reward is calculated.
        crossing = before["cw"] if before is not None else None
        info["crossing_blocked_while_stopped"] = bool(
            crossing and crossing[1] <= 1e-5 and before["v"] < .3)
        if info["crossing_blocked_while_stopped"]:
            reward -= self.drw.get("crossing_stop_penalty", 0.0)
        info["emergency_seen"] = bool(before and before["siren"])
        info["emergency_blocking"] = bool(before and before["siren"] and before["siren"][1])
        info["emergency_closing"] = bool(info["emergency_blocking"] and
                                         before["siren"][0] < 35)
        info["background_vehicles"] = max(0, len(traci.vehicle.getIDList()) - int(self._snap is not None))
        info["front_interaction"] = bool(before and before["gap"] is not None and before["gap"] < 30)
        info["junction_conflict"] = bool(before and before["foe"] and before["foe"][0] < 30)
        info["crossing_occupied"] = bool(before and before["cw"] and before["cw"][2] and before["cw"][1] < 35)
        info["signal_stop"] = bool(before and before["tls"] and before["tls"][2] in RED_STATES
                                   and before["tls"][1] < 12 and before["v"] < .2)
        info["stopped_vehicle"] = bool(before and before["leader_stopped"]
                                       and before["gap"] is not None and before["gap"] < 25)
        info["ego_teleported"] = self.ego_id in traci.simulation.getStartingTeleportIDList()
        info["background_teleports"] = traci.simulation.getStartingTeleportNumber() - int(info["ego_teleported"])
        info["background_collision_vehicles"] = len(set(traci.simulation.getCollidingVehiclesIDList()) - {self.ego_id})
        info["speeding"] = bool(self._snap and self._snap["v"] > self._snap["limit"] + .5)
        if info["speeding"]:
            reward -= .1 * (self._snap["v"] - self._snap["limit"])
        info["unexpected_exit"] = terminated and not any(
            info[k] for k in ("arrived", "collided", "off_route"))
        return obs, reward, terminated or info["ego_teleported"], truncated, info

    def get_privileged_state(self):
        state = super().get_privileged_state()
        if not state["ego_alive"]:
            state["vehicles"] = [
                {"id": vid, "x": x, "y": y, "speed": traci.vehicle.getSpeed(vid),
                 "lane": traci.vehicle.getLaneIndex(vid), "lane_id": traci.vehicle.getLaneID(vid),
                 "road_id": traci.vehicle.getRoadID(vid), "lane_pos": traci.vehicle.getLanePosition(vid)}
                for vid in traci.vehicle.getIDList() for x, y in (traci.vehicle.getPosition(vid),)
            ]
        state["sim_time_s"] = traci.simulation.getTime()
        state["signals"] = {tid: traci.trafficlight.getRedYellowGreenState(tid)
                            for tid in traci.trafficlight.getIDList()}
        state["pedestrians"] = [
            {"id": pid, "position_m": traci.person.getPosition(pid),
             "speed_mps": traci.person.getSpeed(pid), "edge": traci.person.getRoadID(pid)}
            for pid in traci.person.getIDList()
        ]
        return state
