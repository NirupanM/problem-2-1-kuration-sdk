"""`desync/action-motion-lag`: per-window action->motion latency, as a Kuration operator."""
from __future__ import annotations

import numpy as np

from kuration import register_batch_transform, register_operator

MAX_LAG = 15


def _xcorr_lag(action, motion, max_lag=MAX_LAG, valid=None):
    """Lag in ticks maximising corr(action[t], motion[t+lag]); (lag, peak_corr).
    (0, 0.0) means "no evidence": constant action or motion in the window."""
    best_lag, best = 0, 0.0
    a = action - action.mean()
    ok = np.ones(len(a), bool) if valid is None else valid
    for lag in range(0, min(max_lag, len(a) - 8) + 1):
        keep = ok[: len(a) - lag] & ok[lag:]
        x, y = a[: len(a) - lag][keep], motion[lag:][keep]
        if len(x) < 8:
            return 0, 0.0
        denom = x.std() * y.std()
        if denom < 1e-9:
            return 0, 0.0  # constant action or motion in this window: no evidence
        c = float(np.mean((x - x.mean()) * (y - y.mean())) / denom)
        if c > best:
            best_lag, best = lag, c
    return best_lag, best


@register_batch_transform(
    "desync/action-motion-lag",
    outputs=("lag_ticks", "lag_corr"),
    inputs=("field:action", "field:motion"),
    resources={"preferred_batch_size": 64},
)
class ActionMotionLag:
    def transform_batch(self, batch, ctx):
        act = np.asarray(batch[ctx.input_token("field:action")], dtype=np.float64)
        mot = np.asarray(batch[ctx.input_token("field:motion")], dtype=np.float64)
        steps = batch["valid_steps"]
        lag = np.zeros(act.shape[0]); corr = np.zeros(act.shape[0])
        for i, n in enumerate(steps):
            l, c = _xcorr_lag(act[i, :n, 0, 0], mot[i, :n, 0, 0])
            lag[i], corr[i] = l, c
        return {"lag_ticks": lag, "lag_corr": corr}



@register_operator(
    "desync/sync-gate",
    description="Drop windows whose action->motion lag is outside the calibrated band, or unmeasurable",
    parameters={"type": "object", "additionalProperties": False, "properties": {
        "action": {"type": "string"}, "motion": {"type": "string"},
        "max_lag": {"type": "integer", "minimum": 0}, "min_corr": {"type": "number"}},
        "required": ["action", "motion"]},
    version="1",
)
def sync_gate(pipeline, *, action, motion, max_lag=3, min_corr=0.5):
    p = pipeline.transform("desync/action-motion-lag",
                           inputs=[f"field:{action}", f"field:{motion}"])
    return (p
            .filter("lag_corr", ">=", min_corr)      # fail closed: no evidence => no admission
            .filter("lag_ticks", "<=", max_lag))
