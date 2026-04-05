"""
CuedTargetWithDistractorsV2 — NeuroGym environment for a cued detection task.

Observation space shape: (9,)
  [0]     fixation signal
  [1..4]  stimulus channels (one per quadrant)
  [5..8]  cue channels (one per quadrant)

Action space: Discrete(2)
  0 = hold
  1 = release (respond)

Trial structure
---------------
fixation -> cue -> delay (CTOA-driven) -> target -> post_target

CTOA is sampled from a Beta distribution rescaled to [ctoa_range[0], ctoa_range[1]] ms.
Distractors are injected during the delay period with a time-varying hazard rate.
"""

import contextlib

import numpy as np

# NumPy 2.x compatibility patch for older NeuroGym versions
if not hasattr(np, "_no_nep50_warning"):

    @contextlib.contextmanager
    def _no_nep50_warning():
        yield

    np._no_nep50_warning = _no_nep50_warning

import neurogym as ngym
from neurogym import spaces

try:
    from neurogym.core import TrialEnv
except ImportError:
    import neurogym.core as _core

    TrialEnv = _core.TrialEnv


class CuedTargetWithDistractorsV2(TrialEnv):
    """
    Parameters
    ----------
    dt : int
        Simulation timestep in ms.
    rewards : dict or None
        Override default reward values.
    timing : dict or None
        Override default period durations.
    rt_window : tuple[int, int]
        Valid response window (ms) relative to target onset.
    cue_strength : float
        Amplitude of the cue signal (0 = no cue).
    target_strength : float
        Amplitude of the target signal.
    distractor_strength : float
        Amplitude of distractor signals (0 = no distractors).
    distractor_duration : int
        Duration of each distractor in ms.
    p_distractor_trial : float
        Probability that a given trial contains distractors.
    max_distractors : int
        Maximum number of distractors per trial.
    min_gap : int
        Minimum gap between consecutive distractors in ms.
    r_hold : float
        Per-step reward for holding outside the response window.
    time_cost : float
        Per-step penalty applied after target onset.
    fixation_jitter : tuple[int, int]
        Range of fixation period duration in ms.
    cue_duration : int
        Duration of cue in ms.
    post_target_duration : int
        Duration of post-target period in ms.
    ctoa_range : tuple[int, int]
        Range of cue-to-target onset asynchrony in ms.
    ctoa_beta : tuple[float, float]
        Beta distribution shape parameters (a, b) for CTOA sampling.
    distractor_hazard_early : float
        Hazard rate at the start of the delay period.
    distractor_hazard_late : float
        Hazard rate at the end of the delay period.
    allow_fa_overlap_with_target_epoch : bool
        Whether a false-alarm response window can overlap with the target epoch.
    strict_fa_scoring : bool
        If True, responses outside both windows are scored as 'premature'.
    """

    def __init__(
        self,
        dt: int = 20,
        rewards=None,
        timing=None,
        rt_window=(150, 750),
        cue_strength: float = 0.0,
        target_strength: float = 1.0,
        distractor_strength: float = 0.0,
        distractor_duration: int = 100,
        p_distractor_trial: float = 0.0,
        max_distractors: int = 4,
        min_gap: int = 80,
        r_hold: float = 0.0,
        time_cost: float = 0.0,
        fixation_jitter=(300, 600),
        cue_duration: int = 350,
        post_target_duration: int = 900,
        ctoa_range=(1000, 3300),
        ctoa_beta=(2.2, 1.6),
        distractor_hazard_early: float = 0.10,
        distractor_hazard_late: float = 0.01,
        allow_fa_overlap_with_target_epoch: bool = True,
        strict_fa_scoring: bool = True,
    ):
        super().__init__(dt=dt)

        self.rewards = {
            "abort": -1.0,
            "false_alarm": -1.0,
            "correct": 1.0,
            "miss": -1.0,
        }
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
            "delay": 1500,   # placeholder — overwritten each trial
            "target": 100,
            "post_target": self.post_target_duration,
        }
        if timing is not None:
            self.timing.update(timing)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

    # ------------------------------------------------------------------
    # sampling helpers
    # ------------------------------------------------------------------

    def _sample_ctoa_ms(self) -> int:
        lo, hi = self.ctoa_range
        a, b = self.ctoa_beta
        frac = self.rng.beta(a, b)
        return int(lo + frac * (hi - lo))

    def _compute_ctoa_bin(self, ctoa_ms: int) -> int:
        edges = np.linspace(self.ctoa_range[0], self.ctoa_range[1], 11)
        return int(np.digitize([ctoa_ms], edges[1:-1])[0])

    def _sample_distractor_onsets_hazard(
        self, delay_start: int, t_target0: int, dur_steps: int, max_n: int
    ):
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
            p_on = (
                self.distractor_hazard_early * (1.0 - frac)
                + self.distractor_hazard_late * frac
            )

            if self.rng.rand() < p_on:
                onsets.append(int(t))
                blocked_until = t + dur_steps + min_gap_steps

        return onsets

    # ------------------------------------------------------------------
    # trial generation
    # ------------------------------------------------------------------

    def _new_trial(self, **kwargs):
        cue_loc = int(self.rng.choice([1, 2, 3, 4]))
        target_loc = cue_loc
        uncued_locs = [loc for loc in [1, 2, 3, 4] if loc != cue_loc]

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

        # fixation signal (channel 0) — always on
        self.add_ob(
            1.0,
            where=0,
            period=["fixation", "cue", "delay", "target", "post_target"],
        )

        # cue
        if self.cue_strength != 0.0:
            self.add_ob(self.cue_strength, where=4 + cue_loc, period="cue")

        # target
        self.add_ob(self.target_strength, where=target_loc, period="target")

        T = self.ob.shape[0]
        cue_start = self.start_ind["cue"]
        delay_start = self.start_ind["delay"]
        t_target0 = self.start_ind["target"]

        rt0 = int(round(self.rt_window[0] / self.dt))
        rt1 = int(round(self.rt_window[1] / self.dt))
        dur_steps = max(1, int(round(self.distractor_duration / self.dt)))

        # response window mask
        target_win_start = int(np.clip(t_target0 + rt0, 0, T))
        target_win_end = int(np.clip(t_target0 + rt1, 0, T))

        self._resp_window = np.zeros(T, dtype=bool)
        if target_win_end > target_win_start:
            self._resp_window[target_win_start:target_win_end] = True

        self._t_target0 = int(t_target0)
        self._cue_start = int(cue_start)

        # distractor masks and events
        self._distractor_mask = np.zeros(T, dtype=bool)
        self._fa_window = np.zeros(T, dtype=bool)
        self._distractor_events = []

        has_distractors = bool(self.rng.rand() < self.p_distractor_trial)

        if has_distractors and self.distractor_strength != 0.0:
            distractor_onsets = self._sample_distractor_onsets_hazard(
                delay_start=delay_start,
                t_target0=t_target0,
                dur_steps=dur_steps,
                max_n=self.max_distractors,
            )
        else:
            distractor_onsets = []

        distractor_locs = []
        for d0 in distractor_onsets:
            d1 = min(d0 + dur_steps, T)
            d_loc = int(self.rng.choice(uncued_locs))
            distractor_locs.append(d_loc)

            self.ob[d0:d1, d_loc] = self.distractor_strength
            self._distractor_mask[d0:d1] = True

            fa0 = int(np.clip(d0 + rt0, 0, T))
            fa1 = int(np.clip(d0 + rt1, 0, T))
            if not self.allow_fa_overlap_with_target_epoch:
                fa1 = min(fa1, t_target0)

            if fa1 > fa0:
                self._fa_window[fa0:fa1] = True

            self._distractor_events.append(
                {
                    "onset_step": int(d0),
                    "offset_step": int(d1),
                    "loc": int(d_loc),
                    "fa_start": int(fa0),
                    "fa_end": int(fa1),
                    "cdoa_ms": int((d0 - cue_start) * self.dt),
                }
            )

        if distractor_onsets:
            trial["has_distractors"] = True
            trial["n_distractors"] = len(distractor_onsets)
            trial["distractor_locs"] = [int(x) for x in distractor_locs]
            trial["distractor_onsets_steps"] = [int(x) for x in distractor_onsets]
            trial["distractor_cdoa_ms"] = [
                int((d - cue_start) * self.dt) for d in distractor_onsets
            ]
            trial["first_distractor_cdoa_ms"] = int(trial["distractor_cdoa_ms"][0])
            for cdoa in trial["distractor_cdoa_ms"]:
                trial["stimulus_times_ms"].append(int(cdoa))
                trial["stimulus_types"].append("distractor")

        trial["stimulus_times_ms"].append(int(ctoa_ms))
        trial["stimulus_types"].append("target")

        # ground truth
        self.set_groundtruth(0)
        if target_win_end > target_win_start:
            self.gt[target_win_start:target_win_end] = 1

        return trial

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

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

        if (t == self.ob.shape[0] - 1) and not terminated:
            reward += self.rewards["miss"]
            info["train_outcome"] = "miss"
            info["sdt_outcome"] = "miss"
            info["event_type"] = "target"
            terminated = True

        return self.ob_now, float(reward), terminated, False, info
