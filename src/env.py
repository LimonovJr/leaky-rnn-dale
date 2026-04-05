"""
CuedTargetWithDistractorsV3

Observation (7,):
    [0] fixation
    [1] cue_on
    [2] cue_x        — cue position x, scaled by cue_strength
    [3] cue_y        — cue position y, scaled by cue_strength
    [4] stim_on
    [5] stim_x       — stimulus position x, scaled by strength
    [6] stim_y       — stimulus position y, scaled by strength

V3 vs V2: continuous spatial coordinates replace one-hot location channels.
The cue is a near-central object; target/distractors are peripheral.
Same direction identity (loc 1..4) shared by cue and target, different coords.

Action space: Discrete(2) — 0=hold, 1=release
"""

import contextlib
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

    # near-central cue positions (same quadrant as targets, closer to center)
    CUE_POS = {
        1: (-0.40,  0.40),
        2: ( 0.40,  0.40),
        3: (-0.40, -0.40),
        4: ( 0.40, -0.40),
    }

    # peripheral stimulus positions
    STIM_POS = {
        1: (-1.00,  1.00),
        2: ( 1.00,  1.00),
        3: (-1.00, -1.00),
        4: ( 1.00, -1.00),
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

    # ------------------------------------------------------------------ helpers

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

    def _write_cue_epoch(self, cue_loc):
        if self.cue_strength == 0.0:
            return
        t0, t1 = self.start_ind["cue"], self.end_ind["cue"]
        cx, cy = self.CUE_POS[cue_loc]
        self.ob[t0:t1, 1] = 1.0
        self.ob[t0:t1, 2] = self.cue_strength * cx
        self.ob[t0:t1, 3] = self.cue_strength * cy

    def _write_target_epoch(self, target_loc):
        if self.target_strength == 0.0:
            return
        t0, t1 = self.start_ind["target"], self.end_ind["target"]
        sx, sy = self.STIM_POS[target_loc]
        self.ob[t0:t1, 4] = 1.0
        self.ob[t0:t1, 5] = self.target_strength * sx
        self.ob[t0:t1, 6] = self.target_strength * sy

    def _write_distractor_epoch(self, onset_step, offset_step, d_loc):
        if self.distractor_strength == 0.0:
            return
        sx, sy = self.STIM_POS[d_loc]
        self.ob[onset_step:offset_step, 4] = 1.0
        self.ob[onset_step:offset_step, 5] = self.distractor_strength * sx
        self.ob[onset_step:offset_step, 6] = self.distractor_strength * sy

    # ------------------------------------------------------------------ trial

    def _new_trial(self, **kwargs):
        cue_loc = int(self.rng.choice([1, 2, 3, 4]))
        target_loc = cue_loc
        uncued_locs = [l for l in [1, 2, 3, 4] if l != cue_loc]

        ctoa_ms = self._sample_ctoa_ms()
        ctoa_bin = self._compute_ctoa_bin(ctoa_ms)
        self.timing["delay"] = ctoa_ms

        trial = {
            "cue_loc": cue_loc,
            "target_loc": target_loc,
            "valid": True,
            "ctoa_ms": ctoa_ms,
            "ctoa_bin": ctoa_bin,
            "has_distractors": False,
            "n_distractors": 0,
            "distractor_locs": [],
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
        self._write_cue_epoch(cue_loc)
        self._write_target_epoch(target_loc)

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
        for d0 in distractor_onsets:
            d1 = min(d0 + dur_steps, T)
            d_loc = int(self.rng.choice(uncued_locs))
            distractor_locs.append(d_loc)

            self._write_distractor_epoch(d0, d1, d_loc)
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
            trial["distractor_onsets_steps"] = [int(x) for x in distractor_onsets]
            trial["distractor_cdoa_ms"] = [int((d - cue_start) * self.dt) for d in distractor_onsets]
            trial["first_distractor_cdoa_ms"] = int(trial["distractor_cdoa_ms"][0])
            for cdoa in trial["distractor_cdoa_ms"]:
                trial["stimulus_times_ms"].append(int(cdoa))
                trial["stimulus_types"].append("distractor")

        trial["stimulus_times_ms"].append(int(ctoa_ms))
        trial["stimulus_types"].append("target")

        return trial

    # ------------------------------------------------------------------ step

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
