"""Synthetic gameplay-telemetry corpus with *known* ground truth.

Real action-conditioned world-model corpora (CS:GO, Open P2P, ...) mix well-synchronised clips
with clips whose action log is shifted against the video (capture latency, dropped frames,
variable frame rate). We can't ship video, so each clip is the per-tick telemetry that
`pixel_shift` / `slam_reconstruct` would produce:

  forward  action: W held (0/1)                 turn   action: mouse-x direction (-1/0/1)
  speed    measured forward motion              yaw    measured camera yaw rate
  contact  1 on ticks where the player hits a wall (rare; what `collision` would emit)

speed/yaw respond to the actions after `true_lag` ticks. A fraction of clips are DESYNCED
(true_lag 6-10 instead of 1-2). Ground truth is returned separately for evaluation only.
"""
from __future__ import annotations

import numpy as np

from kuration import SchemaSpec, Tick

HZ = 10.0
SCHEMA = SchemaSpec.from_dict({
    "name": "synthetic-gameplay-telemetry",
    "entity_axis": "player",
    "sample_hz": HZ,
    "fields": {
        "forward": {"kind": "scalar"}, "turn": {"kind": "scalar"},
        "speed": {"kind": "scalar"}, "yaw": {"kind": "scalar"},
        "contact": {"kind": "scalar"},
    },
})


def _hold_signal(rng, n, p_switch, values):
    out, cur = np.zeros(n), values[0]
    for i in range(n):
        if rng.random() < p_switch:
            cur = values[rng.integers(len(values))]
        out[i] = cur
    return out


def make_clip(rng, n=600, desynced=False):
    fwd = _hold_signal(rng, n, 0.04, [0.0, 1.0, 1.0])
    turn = _hold_signal(rng, n, 0.06, [0.0, 0.0, -1.0, 1.0])
    lag = int(rng.integers(6, 11)) if desynced else int(rng.integers(1, 3))
    speed, yaw, contact = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        s = t - lag
        speed[t] = 4.0 * fwd[s] if s >= 0 else 0.0
        yaw[t] = 30.0 * turn[s] if s >= 0 else 0.0
    # rare wall hits: while moving forward, speed collapses for 3 ticks (≈1% of ticks)
    for t in np.flatnonzero(speed > 0):
        if rng.random() < 0.004 and t + 3 < n:
            contact[t:t + 3] = 1.0
            speed[t:t + 3] *= 0.05
    speed += rng.normal(0, 0.25, n)
    yaw += rng.normal(0, 2.0, n)
    ticks = [Tick(time=t / HZ, entities={0: {
        "forward": float(fwd[t]), "turn": float(turn[t]), "speed": float(speed[t]),
        "yaw": float(yaw[t]), "contact": float(contact[t])}}) for t in range(n)]
    return ticks, lag


def make_corpus(n_clips=60, desync_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    sources, truth = {}, {}
    for i in range(n_clips):
        desynced = rng.random() < desync_frac
        ticks, lag = make_clip(rng, desynced=desynced)
        sources[f"clip{i:03d}"] = ticks
        truth[f"clip{i:03d}"] = {"lag": lag, "desynced": desynced}
    return sources, truth
