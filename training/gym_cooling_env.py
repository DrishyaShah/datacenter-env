"""
Gymnasium wrapper for the ClusterEnv cooling controller.

Wraps FacilityState directly — no economic layer, no scripted teams, no grader.
The PPO agent learns to control fan speeds and supply temperatures across all 4 zones
given variable IT load schedules generated each episode.

Key training feature: upcoming_load_schedule signal per zone.
This teaches the controller to PRE-COOL zones before a large scheduled job lands —
a behaviour the rule-based heuristic cannot exhibit (it only reacts to current temps).

Observation (25 values, float32):
  Per zone × 4: [temp_c, fan_speed_pct, it_load_kw, supply_air_temp_c, upcoming_load_kw]
  Facility × 5: [outside_temp_c, chiller_cop, chiller_active, carbon_normalized, step_fraction]

Action (8 values, continuous [-1, 1]):
  Per zone × 4: [fan_speed_normalized, supply_setpoint_normalized]
  fan   [-1, 1] → [0, 100] %
  supply [-1, 1] → [16, 26] °C

Episode: PHYSICAL_STEPS_PER_WINDOW (18 steps × 5 sim-minutes = 90 sim-minutes)
"""

from __future__ import annotations
import random
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from server.simulation import (
    FacilityState,
    _DCActionStub,
    _ZoneAdjustmentStub,
)
from server.scenarios.cluster_scenario import (
    build_cluster_facility,
    PHYSICAL_STEPS_PER_WINDOW,
    OUTSIDE_TEMP_SCHEDULE,
    WET_BULB_SCHEDULE,
    CARBON_NUMERIC_SCHEDULE,
    power_budget_violated,
)


# ── Zone ordering (fixed across all episodes) ─────────────────────────────────
ZONE_ORDER = ["zone_team_a_1", "zone_team_a_2", "zone_team_b_1", "zone_shared"]

# ── Load capacity per zone for normalisation ──────────────────────────────────
ZONE_MAX_LOAD = {
    "zone_team_a_1": 480.0,
    "zone_team_a_2": 480.0,
    "zone_team_b_1": 500.0,
    "zone_shared":   300.0,
}

# ── Observation normalisation constants ───────────────────────────────────────
TEMP_MIN, TEMP_RANGE  = 15.0, 30.0     # [15, 45]°C → [0, 1]
LOAD_SCALE            = 600.0          # kW → [0, 1] via /600
SUPPLY_MIN, SUPPLY_R  = 16.0, 10.0    # [16, 26]°C → [0, 1]
OUTSIDE_SCALE         = 45.0          # °C → [0, 1] via /45
COP_SCALE             = 5.0           # [0, 5] → [0, 1]

# ── Action scaling ────────────────────────────────────────────────────────────
FAN_MIN_PCT, FAN_RANGE_PCT       = 0.0,  100.0  # action → fan_speed_pct
SUPPLY_MIN_C, SUPPLY_RANGE_C     = 16.0, 10.0   # action → supply_setpoint

# ── Reward weights ────────────────────────────────────────────────────────────
W_TEMP        = 0.55   # temperature compliance (primary)
W_PUE         = 0.20   # energy efficiency
W_ANTICIPATE  = 0.15   # pre-cooling bonus
W_CARBON      = 0.10   # grid carbon penalty

# Temperature targets
TEMP_SAFE_LO   = 18.0
TEMP_SAFE_HI   = 27.0
TEMP_IDEAL     = 22.0   # reward peaks here
TEMP_PRE_COOL  = 22.0   # must be at or below this to earn anticipation bonus

# Anticipation threshold: upcoming load must be 1.5× current before bonus fires
ANTICIPATION_LOAD_RATIO = 1.5
ANTICIPATION_HORIZON    = 3   # steps ahead to look for incoming load spike


class CoolingGymEnv(gym.Env):
    """
    Gymnasium environment for PPO-based cooling controller pre-training.

    Each episode samples a random IT load trajectory so the controller
    must generalise to loads it hasn't seen — not memorise a fixed curve.
    The upcoming_load_schedule signal enables proactive pre-cooling.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        window_idx: int = 3,
        randomize_loads: bool = True,
        include_chiller_fault: bool = False,
        fault_probability: float = 0.20,
        max_steps: int = PHYSICAL_STEPS_PER_WINDOW,
    ):
        super().__init__()

        self._window_idx        = window_idx        # which window conditions to use
        self._randomize_loads   = randomize_loads   # vary loads between episodes
        self._include_faults    = include_chiller_fault
        self._fault_prob        = fault_probability
        self._max_steps         = max_steps

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        obs_dim = len(ZONE_ORDER) * 5 + 5   # 4 zones × 5 + 5 facility = 25
        self.observation_space = spaces.Box(
            low   = -0.1,
            high  =  1.1,
            shape = (obs_dim,),
            dtype = np.float32,
        )
        # 4 zones × 2 controls (fan, setpoint) = 8 continuous actions in [-1, 1]
        self.action_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (len(ZONE_ORDER) * 2,),
            dtype = np.float32,
        )

        # Runtime state (initialised in reset)
        self.facility:     FacilityState | None = None
        self._last_action: _DCActionStub  | None = None
        self._step:        int                   = 0
        self._load_traj:   dict[str, list[float]] = {}
        self._rng:         random.Random          = random.Random()

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        # Randomise which window conditions to use (temperature + carbon variety)
        w = self._rng.randint(0, 7) if self._randomize_loads else self._window_idx

        # Chiller fault: random chance, triggers at step 9 (mid-episode)
        fault = self._include_faults and self._rng.random() < self._fault_prob
        self.facility = build_cluster_facility(
            seed               = seed,
            window_idx         = w,
            enable_chiller_fault = fault,
            chiller_fault_window = 0,   # fault_step = 0 × 18 = step 9 within episode
        )
        # Translate window-level fault to physical step within THIS episode
        if fault:
            self.facility.chiller_fault_step = self._rng.randint(6, 12)

        self._step = 0
        self._load_traj = self._generate_load_trajectory()

        # Apply step-0 loads
        self.facility.set_all_job_loads(
            {zid: self._load_traj[zid][0] for zid in ZONE_ORDER}
        )
        self.facility.advance_load()

        # Seed last_action from current zone state (no rate-limit delta at step 0)
        self._last_action = self._make_action_stub(
            {z.zone_id: (z.fan_speed_pct, z.supply_air_temp_setpoint_c)
             for z in self.facility.zones}
        )

        return self._obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.facility is not None, "Call reset() before step()"

        # 1. Advance to next step's IT load
        self._step += 1
        if self._step < self._max_steps:
            self.facility.set_all_job_loads(
                {zid: self._load_traj[zid][self._step] for zid in ZONE_ORDER}
            )
        self.facility.advance_load()

        # 2. Decode action → fan% and setpoint per zone
        zone_controls = self._decode_action(action)

        # 3. Create DCActionStub and advance physics
        action_stub   = self._make_action_stub(zone_controls)
        self.facility.step(action_stub, self._last_action)
        self._last_action = action_stub

        # 4. Reward
        reward = self._compute_reward()

        # 5. Termination
        terminated = self._step >= self._max_steps - 1
        truncated  = False

        return self._obs(), float(reward), terminated, truncated, {
            "step": self._step,
            "pue":  self.facility.pue,
            "temps": {z.zone_id: z.temp_c for z in self.facility.zones},
        }

    # ── Observation ───────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        obs = []
        zone_map = {z.zone_id: z for z in self.facility.zones}

        for zid in ZONE_ORDER:
            z        = zone_map[zid]
            upcoming = self._upcoming_load(zid)
            obs.extend([
                (z.temp_c          - TEMP_MIN)   / TEMP_RANGE,
                z.fan_speed_pct    / 100.0,
                z.it_load_kw       / LOAD_SCALE,
                (z.supply_air_temp_c - SUPPLY_MIN) / SUPPLY_R,
                upcoming           / LOAD_SCALE,
            ])

        obs.extend([
            self.facility.outside_temp_c              / OUTSIDE_SCALE,
            self.facility.effective_chiller_cop       / COP_SCALE,
            float(self.facility.chiller_active),
            self.facility.grid_carbon_intensity_normalized,
            self._step / max(self._max_steps - 1, 1),
        ])

        return np.clip(np.array(obs, dtype=np.float32), -0.1, 1.1)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        reward = 0.0
        zone_map = {z.zone_id: z for z in self.facility.zones}

        for zid in ZONE_ORDER:
            z = zone_map[zid]

            # Temperature compliance component
            temp = z.temp_c
            if TEMP_SAFE_LO <= temp <= TEMP_SAFE_HI:
                # Gaussian-shaped reward peaking at TEMP_IDEAL
                proximity = math.exp(-0.5 * ((temp - TEMP_IDEAL) / 2.5) ** 2)
                reward += W_TEMP * proximity
            elif temp > TEMP_SAFE_HI:
                reward -= W_TEMP * min((temp - TEMP_SAFE_HI) / 3.0, 1.0)
            else:
                reward -= W_TEMP * 0.3 * (TEMP_SAFE_LO - temp) / 3.0  # mild undershoot penalty

            # Anticipation bonus: pre-cooled AND large job incoming
            upcoming_kw  = self._upcoming_load(zid)
            current_kw   = z.it_load_kw
            big_job_soon = upcoming_kw > max(current_kw, 50.0) * ANTICIPATION_LOAD_RATIO
            pre_cooled   = temp <= TEMP_PRE_COOL
            if big_job_soon and pre_cooled:
                reward += W_ANTICIPATE

        # PUE efficiency (facility-level, once)
        pue_penalty = (self.facility.pue - 1.0) * W_PUE
        reward -= pue_penalty

        # Carbon penalty when cooling is energy-intensive
        carbon_penalty = (
            self.facility.grid_carbon_intensity_normalized
            * W_CARBON
            * (self.facility.total_fan_power_kw + self.facility.chiller_power_kw)
            / max(self.facility.total_it_load_kw, 1.0)
        )
        reward -= carbon_penalty

        # Normalise by number of zones
        return reward / len(ZONE_ORDER)

    # ── Load trajectory ───────────────────────────────────────────────────────

    def _generate_load_trajectory(self) -> dict[str, list[float]]:
        """
        Generate a per-zone IT load schedule for this episode.
        Three patterns ensure training diversity:
          stable — constant load (tests steady-state efficiency)
          ramp   — load increases over episode (tests gradual response)
          spike  — low early, then a large step increase (tests anticipation)
        """
        trajectories: dict[str, list[float]] = {}
        baselines = {
            "zone_team_a_1": 0.0,
            "zone_team_a_2": 0.0,
            "zone_team_b_1": 180.0,
            "zone_shared":   100.0,
        }

        for zid in ZONE_ORDER:
            base   = baselines[zid]
            max_kw = ZONE_MAX_LOAD[zid]

            if not self._randomize_loads:
                # Fixed moderate load for debugging
                trajectories[zid] = [base + max_kw * 0.4] * self._max_steps
                continue

            pattern = self._rng.choice(["stable", "ramp", "spike"])
            traj    = []

            for s in range(self._max_steps):
                if pattern == "stable":
                    load = self._rng.uniform(0.0, max_kw * 0.65)

                elif pattern == "ramp":
                    frac = s / max(self._max_steps - 1, 1)
                    load = self._rng.uniform(0.0, max_kw * 0.3) + max_kw * 0.55 * frac

                else:  # spike — KEY pattern for anticipation training
                    spike_step = self._rng.randint(
                        self._max_steps // 4,
                        self._max_steps // 2,
                    )
                    if s < spike_step:
                        load = self._rng.uniform(0.0, max_kw * 0.2)   # pre-spike: low
                    else:
                        load = self._rng.uniform(max_kw * 0.55, max_kw * 0.85)  # post-spike: high

                traj.append(round(base + max(0.0, load), 2))

            trajectories[zid] = traj

        return trajectories

    def _upcoming_load(self, zone_id: str) -> float:
        """Average IT load for the next ANTICIPATION_HORIZON steps."""
        future = self._load_traj[zone_id][
            self._step : self._step + ANTICIPATION_HORIZON
        ]
        return sum(future) / max(len(future), 1) if future else 0.0

    # ── Action decoding ───────────────────────────────────────────────────────

    def _decode_action(self, action: np.ndarray) -> dict[str, tuple[float, float]]:
        """Map flat [-1, 1] action array → {zone_id: (fan_pct, supply_setpoint)}."""
        controls = {}
        for i, zid in enumerate(ZONE_ORDER):
            fan_raw     = action[i * 2]
            supply_raw  = action[i * 2 + 1]
            fan_pct     = float(np.clip((fan_raw + 1.0) / 2.0 * 100.0, 0.0, 100.0))
            supply_c    = float(np.clip((supply_raw + 1.0) / 2.0 * 10.0 + 16.0, 16.0, 26.0))
            controls[zid] = (fan_pct, supply_c)
        return controls

    # ── Stub builders ─────────────────────────────────────────────────────────

    def _make_action_stub(
        self, zone_controls: dict[str, tuple[float, float]]
    ) -> _DCActionStub:
        return _DCActionStub(
            zone_adjustments=[
                _ZoneAdjustmentStub(
                    zone_id                    = zid,
                    fan_speed_pct              = ctrl[0],
                    supply_air_temp_setpoint_c = ctrl[1],
                )
                for zid, ctrl in zone_controls.items()
            ],
            chiller_setpoint_c = 10.0,
            chiller_active     = self.facility.chiller_active,
        )
