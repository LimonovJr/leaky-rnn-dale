"""
CuedTargetWithDistractorsV3

Observation (7,):
    [0] fixation
    [1] cue_x         — cue x-coordinate (raw, not scaled)
    [2] cue_y         — cue y-coordinate (raw)
    [3] cue_strength  — cue amplitude (=cue_strength in cue epoch, else 0)
    [4] stim_x        — stimulus x-coordinate (target or distractor)
    [5] stim_y        — stimulus y-coordinate
    [6] stim_strength — stimulus amplitude (target_strength / distractor_strength
                        in the corresponding epoch, else 0)

The cue is a near-central object; target/distractors are peripheral. Same
direction identity (loc 1..4) is shared by cue and target. Channels 1-3 and
4-6 are consumed by the geometric Gaussian RF in BioLeakyRNNTopo (Chen &
Gong 2022, Eq. 35). On-markers (cue_on, stim_on) are NOT used — the
`strength` channels encode presence implicitly (=0 when absent).

Action space: Discrete(2) — 0=hold, 1=release
"""

import contextlib
import math

import numpy as np

if not hasattr(np, "_no_nep50_warning"):
    @contextlib.contextmanager
    def _no_nep50_warning():
        yield
    np._no_nep50_warning = _no_nep50_warning

import neurogym as ngym  # noqa: F401
from neurogym import spaces

try:
    from neurogym.core import TrialEnv
except ImportError:
    import neurogym.core as _core
    TrialEnv = _core.TrialEnv


class CuedTargetWithDistractorsV3(TrialEnv):

    # near-central cue positions (same quadrant as targets, closer to center).
    # Wider than earlier versions (was +-0.10) so that cue identities are
    # resolvable under the Gaussian RF input (rf_sigma=0.3) in BioLeakyRNNTopo.
    CUE_POS = {
        1: (-0.40,  0.40),
        2: ( 0.40,  0.40),
        3: (-0.40, -0.40),
        4: ( 0.40, -0.40),
    }

    # peripheral stimulus positions
    # NOTE: kept inside the sheet (coords in [-1, +1]) so the Gaussian RF
    # drive (sigma=0.3) is not truncated at the edge. With +-0.7 there is
    # ~1 sigma of buffer on each side, giving all 4 corners a symmetric,
    # equal-strength drive profile.
    STIM_POS = {
        1: (-0.70,  0.70),
        2: ( 0.70,  0.70),
        3: (-0.70, -0.70),
        4: ( 0.70, -0.70),
    }

    def __init__(
        self,
        dt=20,
        rewards=None,
        timing=None,
        rt_window=(150, 750),
        cue_strength=1.0,
        target_strength=1.0,
        distractor_strength=1.0,
        distractor_duration=100,
        p_distractor_trial=0.6,
        max_distractors=4,
        min_gap=80,
        r_hold=0.0,
        time_cost=0.0,
        fixation_jitter=(700, 1200),
        cue_duration=350,
        post_target_duration=900,
        ctoa_range=(1000, 3300),
        ctoa_beta=(2.2, 1.6),
        distractor_hazard_early=0.10,
        distractor_hazard_late=0.01,
        allow_fa_overlap_with_target_epoch=True,
        strict_fa_scoring=True,
        # --- continuous-location mode (opt-in; default False preserves legacy behaviour) ---
        continuous_locations=False,
        cue_radius=0.4,
        target_radius=0.7,
        distractor_min_sep_rad=math.pi / 3,
    ):
        super().__init__(dt=dt)

        self.rewards = {"abort": -1.0, "false_alarm": -1.0, "correct": 1.0, "miss": -1.0}
        if rewards is not None:
            self.rewards.update(rewards)

        self.rt_window = tuple(rt_window)
        self.cue_strength = float(cue_strength)
        self.target_strength = float(target_strength)
        self.distractor_strength = float(distractor_strength)
        self.distractor_duration = int(distractor_duration)
        self.p_distractor_trial = float(p_distractor_trial)
        self.max_distractors = int(max_distractors)
        self.min_gap = int(min_gap)
        self.r_hold = float(r_hold)
        self.time_cost = float(time_cost)
        self.fixation_jitter = tuple(fixation_jitter)
        self.cue_duration = int(cue_duration)
        self.post_target_duration = int(post_target_duration)
        self.ctoa_range = tuple(ctoa_range)
        self.ctoa_beta = tuple(ctoa_beta)
        self.distractor_hazard_early = float(distractor_hazard_early)
        self.distractor_hazard_late = float(distractor_hazard_late)
        self.allow_fa_overlap_with_target_epoch = bool(allow_fa_overlap_with_target_epoch)
        self.strict_fa_scoring = bool(strict_fa_scoring)

        self.continuous_locations     = bool(continuous_locations)
        self.cue_radius               = float(cue_radius)
        self.target_radius            = float(target_radius)
        self.distractor_min_sep_rad   = float(distractor_min_sep_rad)

        self.timing = {
            "fixation": self.fixation_jitter,
            "cue": self.cue_duration,
            "delay": 1500,   # overwritten each trial
            "target": 100,
            "post_target": self.post_target_duration,
        }
        if timing is not None:
            self.timing.update(timing)

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def _sample_ctoa_ms(self):
        lo, hi = self.ctoa_range
        a, b = self.ctoa_beta
        return int(lo + self.rng.beta(a, b) * (hi - lo))

    def _compute_ctoa_bin(self, ctoa_ms):
        edges = np.linspace(self.ctoa_range[0], self.ctoa_range[1], 11)
        return int(np.digitize([ctoa_ms], edges[1:-1])[0])

    def _sample_distractor_onsets_hazard(self, delay_start, t_target0, dur_steps, max_n):
        onsets = []
        blocked_until = -1
        min_gap_steps = max(1, int(round(self.min_gap / self.dt)))

        if self.allow_fa_overlap_with_target_epoch:
            latest_onset = max(delay_start, t_target0 - dur_steps - 1)
        else:
            rt1 = int(round(self.rt_window[1] / self.dt))
            latest_onset = max(delay_start, t_target0 - rt1 - dur_steps)

        if latest_onset <= delay_start:
            return onsets

        denom = max(1, latest_onset - delay_start)
        for t in range(delay_start, latest_onset + 1):
            if len(onsets) >= max_n:
                break
            if t <= blocked_until:
                continue
            frac = (t - delay_start) / denom
            p_on = self.distractor_hazard_early * (1 - frac) + self.distractor_hazard_late * frac
            if self.rng.rand() < p_on:
                onsets.append(int(t))
                blocked_until = t + dur_steps + min_gap_steps

        return onsets

    def _set_fixation_channel(self):
        self.ob[:, 0] = 1.0

    def _write_cue_epoch(self, cue_xy):
        """cue_xy: (cx, cy) position tuple."""
        if self.cue_strength == 0.0:
            return
        t0, t1 = self.start_ind["cue"], self.end_ind["cue"]
        cx, cy = cue_xy
        self.ob[t0:t1, 1] = cx
        self.ob[t0:t1, 2] = cy
        self.ob[t0:t1, 3] = self.cue_strength

    def _write_target_epoch(self, target_xy):
        """target_xy: (sx, sy) position tuple."""
        if self.target_strength == 0.0:
            return
        t0, t1 = self.start_ind["target"], self.end_ind["target"]
        sx, sy = target_xy
        self.ob[t0:t1, 4] = sx
        self.ob[t0:t1, 5] = sy
        self.ob[t0:t1, 6] = self.target_strength

    def _write_distractor_epoch(self, onset_step, offset_step, d_xy):
        """d_xy: (sx, sy) position tuple."""
        if self.distractor_strength == 0.0:
            return
        sx, sy = d_xy
        self.ob[onset_step:offset_step, 4] = sx
        self.ob[onset_step:offset_step, 5] = sy
        self.ob[onset_step:offset_step, 6] = self.distractor_strength

    def _sample_cue_target_theta(self):
        """Uniform angle on the ring, θ ∈ [0, 2π)."""
        return float(self.rng.uniform(0.0, 2.0 * math.pi))

    def _sample_distractor_theta(self, cue_theta, max_tries=64):
        """
        Sample θ_d ∈ [0, 2π) with wrapped angular distance to cue_theta
        at least self.distractor_min_sep_rad. Falls back to the point farthest
        from cue_theta if rejection sampling fails after max_tries (shouldn't
        happen with min_sep < π).
        """
        two_pi = 2.0 * math.pi
        for _ in range(max_tries):
            t = float(self.rng.uniform(0.0, two_pi))
            d = abs((t - cue_theta + math.pi) % two_pi - math.pi)
            if d >= self.distractor_min_sep_rad:
                return t
        # fallback: opposite side
        return (cue_theta + math.pi) % two_pi

    @staticmethod
    def _theta_to_quadrant(theta):
        """
        Map angle to legacy 1-4 location index. Convention aligned with
        CUE_POS/STIM_POS:  1:(-,+), 2:(+,+), 3:(-,-), 4:(+,-).
        """
        x = math.cos(theta); y = math.sin(theta)
        if   x <  0 and y >= 0: return 1
        elif x >= 0 and y >= 0: return 2
        elif x <  0 and y <  0: return 3
        else:                   return 4

    def _sample_positions(self):
        """Return (cue_pos, target_pos, cue_theta, target_theta, cue_loc,
        target_loc, uncued_locs). Overridable hook so subclasses can change the
        spatial layout (e.g. full-2D same-location) without duplicating the rest
        of trial construction. Default = ring layout (this class's behaviour)."""
        if self.continuous_locations:
            cue_theta = self._sample_cue_target_theta()
            target_theta = cue_theta   # target is co-located with cue on the ring
            cue_pos    = (self.cue_radius    * math.cos(cue_theta),
                          self.cue_radius    * math.sin(cue_theta))
            target_pos = (self.target_radius * math.cos(target_theta),
                          self.target_radius * math.sin(target_theta))
            # back-compat indices (nearest of 4 quadrants) for code that buckets
            # trials by loc; the authoritative info is in cue_pos/target_pos/θ.
            cue_loc    = self._theta_to_quadrant(cue_theta)
            target_loc = cue_loc
            uncued_locs = None  # not used in continuous mode
        else:
            cue_loc    = int(self.rng.choice([1, 2, 3, 4]))
            target_loc = cue_loc
            uncued_locs = [l for l in [1, 2, 3, 4] if l != cue_loc]
            cue_pos    = self.CUE_POS[cue_loc]
            target_pos = self.STIM_POS[target_loc]
            # derive θ from the discrete position for a uniform trial interface
            cue_theta    = math.atan2(cue_pos[1],    cue_pos[0])
            target_theta = math.atan2(target_pos[1], target_pos[0])
        return (cue_pos, target_pos, cue_theta, target_theta,
                cue_loc, target_loc, uncued_locs)

    def _new_trial(self, **kwargs):
        # location sampling (overridable via _sample_positions)
        (cue_pos, target_pos, cue_theta, target_theta,
         cue_loc, target_loc, uncued_locs) = self._sample_positions()

        ctoa_ms = self._sample_ctoa_ms()
        ctoa_bin = self._compute_ctoa_bin(ctoa_ms)
        self.timing["delay"] = ctoa_ms

        trial = {
            "cue_loc": cue_loc,
            "target_loc": target_loc,
            "cue_pos": cue_pos,
            "target_pos": target_pos,
            "cue_theta": cue_theta,
            "target_theta": target_theta,
            "valid": True,
            "ctoa_ms": ctoa_ms,
            "ctoa_bin": ctoa_bin,
            "has_distractors": False,
            "n_distractors": 0,
            "distractor_locs": [],
            "distractor_positions": [],
            "distractor_thetas": [],
            "distractor_onsets_steps": [],
            "distractor_cdoa_ms": [],
            "first_distractor_cdoa_ms": None,
            "stimulus_times_ms": [],
            "stimulus_types": [],
        }
        trial.update(kwargs)

        self.add_period(["fixation", "cue", "delay", "target", "post_target"])

        T = self.end_ind["post_target"]
        self.ob = np.zeros((T, 7), dtype=np.float32)
        self.gt = np.zeros(T, dtype=np.int64)

        self._set_fixation_channel()
        self._write_cue_epoch(cue_pos)
        self._write_target_epoch(target_pos)

        cue_start   = self.start_ind["cue"]
        delay_start = self.start_ind["delay"]
        t_target0   = self.start_ind["target"]

        rt0 = int(round(self.rt_window[0] / self.dt))
        rt1 = int(round(self.rt_window[1] / self.dt))
        dur_steps = max(1, int(round(self.distractor_duration / self.dt)))

        win_start = int(np.clip(t_target0 + rt0, 0, T))
        win_end   = int(np.clip(t_target0 + rt1, 0, T))

        self._resp_window = np.zeros(T, dtype=bool)
        if win_end > win_start:
            self._resp_window[win_start:win_end] = True
            self.gt[win_start:win_end] = 1

        self._t_target0 = int(t_target0)
        self._cue_start = int(cue_start)

        self._distractor_mask = np.zeros(T, dtype=bool)
        self._fa_window = np.zeros(T, dtype=bool)
        self._distractor_events = []

        distractors_enabled = (
            self.p_distractor_trial > 0.0
            and self.distractor_strength != 0.0
            and self.max_distractors > 0
        )

        if distractors_enabled and self.rng.rand() < self.p_distractor_trial:
            distractor_onsets = self._sample_distractor_onsets_hazard(
                delay_start, t_target0, dur_steps, self.max_distractors
            )
        else:
            distractor_onsets = []

        distractor_locs = []
        distractor_positions = []
        distractor_thetas = []
        for d0 in distractor_onsets:
            d1 = min(d0 + dur_steps, T)

            # Geometric "not-near-cue" criterion:
            # - discrete mode: pick from the 3 non-cued corners (angular separation ≥ π/2)
            # - continuous mode: sample θ_d with |Δθ| ≥ distractor_min_sep_rad from cue
            if self.continuous_locations:
                d_theta = self._sample_distractor_theta(cue_theta)
                d_pos   = (self.target_radius * math.cos(d_theta),
                           self.target_radius * math.sin(d_theta))
                d_loc   = self._theta_to_quadrant(d_theta)  # back-compat quadrant
            else:
                d_loc   = int(self.rng.choice(uncued_locs))
                d_pos   = self.STIM_POS[d_loc]
                d_theta = math.atan2(d_pos[1], d_pos[0])

            distractor_locs.append(d_loc)
            distractor_positions.append(d_pos)
            distractor_thetas.append(d_theta)

            self._write_distractor_epoch(d0, d1, d_pos)
            self._distractor_mask[d0:d1] = True

            fa0 = int(np.clip(d0 + rt0, 0, T))
            fa1 = int(np.clip(d0 + rt1, 0, T))
            if not self.allow_fa_overlap_with_target_epoch:
                fa1 = min(fa1, t_target0)
            if fa1 > fa0:
                self._fa_window[fa0:fa1] = True

            self._distractor_events.append({
                "onset_step": int(d0),
                "offset_step": int(d1),
                "loc": int(d_loc),
                "fa_start": int(fa0),
                "fa_end": int(fa1),
                "cdoa_ms": int((d0 - cue_start) * self.dt),
            })

        if distractor_onsets:
            trial["has_distractors"] = True
            trial["n_distractors"] = len(distractor_onsets)
            trial["distractor_locs"] = [int(x) for x in distractor_locs]
            trial["distractor_positions"] = [
                (float(p[0]), float(p[1])) for p in distractor_positions
            ]
            trial["distractor_thetas"] = [float(t) for t in distractor_thetas]
            trial["distractor_onsets_steps"] = [int(x) for x in distractor_onsets]
            trial["distractor_cdoa_ms"] = [int((d - cue_start) * self.dt) for d in distractor_onsets]
            trial["first_distractor_cdoa_ms"] = int(trial["distractor_cdoa_ms"][0])
            for cdoa in trial["distractor_cdoa_ms"]:
                trial["stimulus_times_ms"].append(int(cdoa))
                trial["stimulus_types"].append("distractor")

        trial["stimulus_times_ms"].append(int(ctoa_ms))
        trial["stimulus_types"].append("target")

        return trial

    def _step(self, action):
        t = self.t_ind
        in_target_window = bool(self._resp_window[t])
        distractor_on_now = bool(self._distractor_mask[t])

        active_fa_event = None
        for ev in self._distractor_events:
            if ev["fa_start"] <= t < ev["fa_end"]:
                active_fa_event = ev
                break
        in_fa_window = active_fa_event is not None

        reward = 0.0
        terminated = False

        info = {
            "gt": int(self.gt_now),
            "cue_loc": int(self.trial["cue_loc"]),
            "target_loc": int(self.trial["target_loc"]),
            "valid": True,
            "ctoa_ms": int(self.trial["ctoa_ms"]),
            "ctoa_bin": int(self.trial["ctoa_bin"]),
            "has_distractors": bool(self.trial["has_distractors"]),
            "n_distractors": int(self.trial["n_distractors"]),
            "distractor_locs": list(self.trial["distractor_locs"]),
            "distractor_cdoa_ms": list(self.trial["distractor_cdoa_ms"]),
            "first_distractor_cdoa_ms": self.trial["first_distractor_cdoa_ms"],
            "distractor_on_now": distractor_on_now,
            "in_fa_window": in_fa_window,
            "in_target_window": in_target_window,
            "active_fa_loc": int(active_fa_event["loc"]) if active_fa_event else None,
            "active_fa_onset_ms": int(active_fa_event["cdoa_ms"]) if active_fa_event else None,
            "train_outcome": None,
            "sdt_outcome": None,
            "event_type": None,
            "rt_ms": None,
        }

        if action == 0 and self.r_hold and not in_target_window:
            reward += self.r_hold
        if self.time_cost and t >= self._t_target0:
            reward -= self.time_cost

        if action == 1:
            if in_target_window:
                reward += self.rewards["correct"]
                info["train_outcome"] = "correct"
                info["sdt_outcome"] = "hit"
                info["event_type"] = "target"
                info["rt_ms"] = int((t - self._t_target0) * self.dt)
                terminated = True
            elif in_fa_window:
                reward += self.rewards["false_alarm"]
                info["train_outcome"] = "false_alarm"
                info["sdt_outcome"] = "false_alarm"
                info["event_type"] = "distractor"
                info["rt_ms"] = int((t - active_fa_event["onset_step"]) * self.dt)
                terminated = True
            else:
                reward += self.rewards["abort"]
                info["train_outcome"] = "abort"
                info["sdt_outcome"] = "premature" if self.strict_fa_scoring else "false_alarm"
                info["event_type"] = "none"
                terminated = True

        if t == self.ob.shape[0] - 1 and not terminated:
            reward += self.rewards["miss"]
            info["train_outcome"] = "miss"
            info["sdt_outcome"] = "miss"
            info["event_type"] = "target"
            terminated = True

        return self.ob_now, float(reward), terminated, False, info


class SpatialPretrain(TrialEnv):
    """Spatial pretraining env: a cue at a random continuous (x,y) on for the
    whole trial; no fixation/cue/target epochs, no distractors, no response
    window. 7-channel obs identical in layout to CuedTargetWithDistractorsV3
    (cue is encoded via channels 1-3; stim channels stay zero) so that
    pretrained weights transfer cleanly to the main task. Cue (x,y) is exposed
    via self.trial["target_pos"] so rollout_one_trial picks it up as the
    MSE regression target.
    """

    def __init__(self, dt=20, trial_len_ms=1500, cue_range=(-1.0, 1.0),
                 cue_strength=1.0):
        super().__init__(dt=dt)
        self.trial_len_ms = int(trial_len_ms)
        self.cue_range = (float(cue_range[0]), float(cue_range[1]))
        self.cue_strength = float(cue_strength)

        self.timing = {"cue": self.trial_len_ms}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,),
                                            dtype=np.float32)
        # 2D continuous (x, y) output target during pretraining. Reward path
        # is unused; supervised training only reads self.ob / self.gt /
        # self.trial via rollout_one_trial.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _new_trial(self, **kwargs):
        lo, hi = self.cue_range
        cx = float(self.rng.uniform(lo, hi))
        cy = float(self.rng.uniform(lo, hi))

        trial = {"cue_pos": (cx, cy), "target_pos": (cx, cy)}
        trial.update(kwargs)

        self.add_period(["cue"])
        T = self.end_ind["cue"]

        self.ob = np.zeros((T, 7), dtype=np.float32)
        self.gt = np.zeros(T, dtype=np.int64)

        self.ob[:, 0] = 1.0                 # fixation always on
        self.ob[:, 1] = cx
        self.ob[:, 2] = cy
        self.ob[:, 3] = self.cue_strength
        # channels 4-6 (stim) stay zero — no stimulus in pretraining
        return trial

    def _step(self, action):
        t = self.t_ind
        terminated = bool(t >= self.ob.shape[0] - 1)
        info = {"gt": int(self.gt[t]),
                "cue_pos": self.trial["cue_pos"]}
        return self.ob[t], 0.0, terminated, False, info


class CuedTargetSpatialV3(CuedTargetWithDistractorsV3):
    """Spatial-output analog of CuedTargetWithDistractorsV3.

    Same trial structure (fixation → cue → delay → target → post-target,
    optional distractors, continuous_locations support) but the action
    space is continuous 2-D (x, y) instead of binary hold/release. The
    network is supervised to output target_pos throughout the trial; the
    outcome is "correct" iff the mean distance to target_pos inside the
    response window is below `correct_threshold`. Distractor capture is
    flagged when mean distance to a distractor's position is below
    `distractor_pull_threshold` inside that distractor's FA window.

    Observation, epoch geometry, distractor sampling — inherited unchanged.
    """

    def __init__(self, *args,
                 correct_threshold: float = 0.15,
                 distractor_pull_threshold: float = 0.15,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # New continuous action space — overrides the parent's Discrete(2).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,),
                                       dtype=np.float32)
        self.correct_threshold = float(correct_threshold)
        self.distractor_pull_threshold = float(distractor_pull_threshold)
        self._step_actions = None          # per-trial buffer of (t, ax, ay)
        self._target_pos_arr = None        # np.array([tx, ty]) for fast _step
        self._response_window = (0, 0)     # (win_start, win_end) step indices

    def _new_trial(self, **kwargs):
        trial = super()._new_trial(**kwargs)

        tx, ty = trial["target_pos"]
        self._target_pos_arr = np.array([tx, ty], dtype=np.float32)
        self._step_actions = []  # list of (t, ax, ay) per step

        # Response window mirrors the binary rt_window relative to target onset.
        rt0 = int(round(self.rt_window[0] / self.dt))
        rt1 = int(round(self.rt_window[1] / self.dt))
        T = self.ob.shape[0]
        win_start = int(np.clip(self._t_target0 + rt0, 0, T))
        win_end   = int(np.clip(self._t_target0 + rt1, 0, T))
        self._response_window = (win_start, win_end)

        # Multi-task: KEEP the parent's hold/release gt (1 inside the response
        # window, 0 elsewhere) so the timing head can be supervised by CE. The
        # (x,y) position head uses target_pos; this gt drives the "when" head.
        return trial

    def _step(self, action):
        t = self.t_ind

        # Action is a 2-D continuous (x, y). Numpy or torch tensor both work.
        ax = float(action[0])
        ay = float(action[1])
        tx, ty = float(self._target_pos_arr[0]), float(self._target_pos_arr[1])
        distance = float(np.hypot(ax - tx, ay - ty))
        self._step_actions.append((int(t), ax, ay))

        # Small per-step penalty so reward is meaningful for any RL diagnostics.
        # Supervised training doesn't use it.
        reward = -distance * (self.dt / 1000.0)
        terminated = bool(t >= self.ob.shape[0] - 1)

        info = {
            "gt": 0,  # placeholder; spatial task has no class target
            "cue_loc": int(self.trial["cue_loc"]),
            "target_loc": int(self.trial["target_loc"]),
            "target_pos": tuple(self.trial["target_pos"]),
            "cue_pos": tuple(self.trial["cue_pos"]),
            "cue_theta": float(self.trial["cue_theta"]),
            "target_theta": float(self.trial["target_theta"]),
            "ctoa_ms": int(self.trial["ctoa_ms"]),
            "ctoa_bin": int(self.trial["ctoa_bin"]),
            "has_distractors": bool(self.trial["has_distractors"]),
            "n_distractors": int(self.trial["n_distractors"]),
            "distractor_locs": list(self.trial["distractor_locs"]),
            "distractor_positions": list(self.trial["distractor_positions"]),
            "distractor_cdoa_ms": list(self.trial["distractor_cdoa_ms"]),
            "first_distractor_cdoa_ms": self.trial["first_distractor_cdoa_ms"],
            "distance_now": distance,
            # Outcome fields populated only at trial end (below).
            "train_outcome": None,
            "mean_err": None,
            "distractor_pull": None,
            "rt_ms": None,
        }

        if terminated:
            self._finalize_outcome(info)

        return self.ob[t], float(reward), terminated, False, info

    def _finalize_outcome(self, info):
        tx = float(self._target_pos_arr[0])
        ty = float(self._target_pos_arr[1])
        # Distance to TARGET per step, indexed by t for fast lookup.
        d_target = {ti: float(np.hypot(ax - tx, ay - ty))
                    for (ti, ax, ay) in self._step_actions}

        ws, we = self._response_window
        dists_in_window = [d for ti, d in d_target.items() if ws <= ti < we]
        mean_err = float(np.mean(dists_in_window)) if dists_in_window else float("nan")

        rt_ms = None
        for ti, _, _ in self._step_actions:
            if ti >= self._t_target0 and d_target[ti] < self.correct_threshold:
                rt_ms = int((ti - self._t_target0) * self.dt)
                break

        # Distractor pull: for each distractor, mean distance from action to
        # THAT distractor's position inside its FA window. Below threshold ⇒
        # model was captured.
        distractor_pull = False
        for ev, d_pos in zip(self._distractor_events,
                             self.trial["distractor_positions"]):
            dx, dy = float(d_pos[0]), float(d_pos[1])
            fa_start, fa_end = ev["fa_start"], ev["fa_end"]
            d_dist = [float(np.hypot(ax - dx, ay - dy))
                      for (ti, ax, ay) in self._step_actions
                      if fa_start <= ti < fa_end]
            if d_dist and float(np.mean(d_dist)) < self.distractor_pull_threshold:
                distractor_pull = True
                break

        if not np.isnan(mean_err) and mean_err < self.correct_threshold:
            outcome = "correct"
        elif distractor_pull:
            outcome = "false_alarm"
        else:
            outcome = "miss"

        info["train_outcome"] = outcome
        info["mean_err"] = float(mean_err) if not np.isnan(mean_err) else None
        info["distractor_pull"] = bool(distractor_pull)
        info["rt_ms"] = rt_ms


class CuedTargetSameLoc2D(CuedTargetWithDistractorsV3):
    """Go/no-go env where the cue and target appear at the SAME location, sampled
    uniformly from the full 2D map (a disk of radius <= target_radius) rather than
    on a fixed ring. Used to study spatial working memory: the network must hold a
    single 2D location across the delay.

    Action space and outcome logic are the inherited binary hold/release; the only
    change is the spatial layout (cue_pos == target_pos at a random 2D point), via
    the _sample_positions hook. Continuous-by-construction (ignores
    continuous_locations / cue_radius — cue and target share one position).
    """

    def _sample_positions(self):
        # Uniform over the disk AREA: r = R*sqrt(U) gives positions that fill the
        # 2D map rather than clustering near the centre. cue and target co-located.
        R = self.target_radius
        r = R * math.sqrt(self.rng.rand())
        theta = 2.0 * math.pi * self.rng.rand()
        pos = (r * math.cos(theta), r * math.sin(theta))
        cue_loc = self._theta_to_quadrant(theta)
        return (pos, pos, theta, theta, cue_loc, cue_loc, None)
