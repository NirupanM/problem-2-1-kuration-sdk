"""Sync-aware curation for action-conditioned world models, built on the Kuration SDK.

    python src/main.py [--clips 60] [--n 120] [--out out/]

Three arms are curated from the same corpus (same budget, same seed) and scored against hidden
ground truth and a downstream controllability proxy:

  naive     .sample("equal")                                    what you get with no curation
  stock     rare_event_coverage + balanced_selector             SDK recipes only
  sync(window) / sync(clip)   our desync/sync-gate operator, per window or per clip, then rare_event_coverage
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import desync  # noqa: F401  (registers desync/* components)
from synth import SCHEMA, make_corpus

import kuration.selection  # noqa: F401  (registers the stock recipes as Pipeline methods)
from kuration import input


def _windows_fsync_shim():
    """SDK gap #1: io._fsync_file opens O_RDONLY, and os.fsync on a read-only fd raises EBADF on
    Windows, so .output() cannot run there. Patched here instead of editing the SDK."""
    import os
    from kuration.core import io as kio

    def fsync_file(path):
        fd = os.open(path, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    kio._fsync_file = fsync_file


if sys.platform == "win32":
    _windows_fsync_shim()


def base(sources):
    return (input(sources, schema=SCHEMA).window(64, stride=32)
            .measure("turn_activity", "std", of="turn")
            .measure("cr", "rate", of="contact"))  # evaluation-only copy of the event rate


def arm_naive(p, n):
    return p  # no selection: main() draws uniformly at random from all candidate windows


def arm_stock(p, n):
    return p.rare_event_coverage(of="contact", feature="contact_rate", count=n)


def arm_sync(p, n):
    gated = p.sync_gate(action="forward", motion="speed", max_lag=3, min_corr=0.6)
    return gated.rare_event_coverage(of="contact", feature="contact_rate", count=n)


def clip_level_sources(sources, max_lag=3, min_corr=0.6):
    """Window gates are blind when the action is held for a whole window (no onset to correlate).
    Workaround for the missing clip-level aggregation: pass 1 estimates lag per window, we take
    each clip's median over *confident* windows, and pass 2 runs on the admitted clips only."""
    wins = (input(sources, schema=SCHEMA).window(64, stride=32)
            .transform("desync/action-motion-lag", inputs=["field:forward", "field:speed"])
            .collect(n=None).selected)
    per_clip = {}
    for s in wins:
        if s.features["lag_corr"] >= min_corr:
            per_clip.setdefault(s.source_id, []).append(s.features["lag_ticks"])
    ok = {c for c, v in per_clip.items() if np.median(v) <= max_lag}
    return {c: t for c, t in sources.items() if c in ok}


def evaluate(name, result, sources, truth, holdout):
    sel = result.selected
    desynced = np.mean([truth[s.source_id]["desynced"] for s in sel]) if sel else float("nan")
    contact = np.mean([s.features["cr"] > 0 for s in sel]) if sel else 0.0
    # Downstream proxy -- "causal sharpness": regress speed_t on forward_{t-k}, k=0..K over the
    # selected windows (what any action-conditioned model must learn). Mass of the learned
    # kernel on the true latency (1-2 ticks) is high when action->effect is consistent and smears
    # out when desynced clips are mixed in.
    K = 12
    rows, ys = [], []
    for s in sel:
        ticks = sources[s.source_id][s.start_idx:s.end_idx]
        f = np.array([t.entities[0]["forward"] for t in ticks]); v = np.array([t.entities[0]["speed"] for t in ticks])
        for t in range(K, len(f)):
            rows.append(f[t - K:t + 1][::-1]); ys.append(v[t])
    kernel = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)[0]
    sharp = float(np.abs(kernel[1:3]).sum() / np.abs(kernel).sum())
    return {"arm": name, "windows": len(sel), "pct_desynced": round(100 * float(desynced), 1),
            "pct_with_contact": round(100 * float(contact), 1), "causal_sharpness": round(sharp, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=60)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    sources, truth = make_corpus(a.clips, seed=0)
    clean_src, clean_truth = make_corpus(20, desync_frac=0.0, seed=99)
    f = np.concatenate([[t.entities[0]["forward"] for t in tk][:-1] for tk in clean_src.values()])
    v = np.concatenate([[t.entities[0]["speed"] for t in tk][1:] for tk in clean_src.values()])
    holdout = (f, v)

    arms = {"naive": arm_naive, "stock": arm_stock, "sync(window)": arm_sync, "sync(clip)": arm_stock}
    rows = []
    for name, fn in arms.items():
        pipe = fn(base(clip_level_sources(sources) if name == "sync(clip)" else sources), a.n).options(seed=7)
        if name == "sync(clip)":
            print(pipe.explain().render().split("warning")[0])
            pipe.output(str(Path(a.out) / "sync_curated"), force=True)  # shards + config.json (reproducible)
        res = pipe.collect(n=None) if name == "naive" else pipe.collect()
        if name == "naive":  # uniform random draw from every candidate window
            from types import SimpleNamespace
            idx = np.random.default_rng(7).choice(len(res.selected), a.n, replace=False)
            res = SimpleNamespace(selected=[res.selected[i] for i in idx])
        rows.append(evaluate(name, res, sources, truth, holdout))
    print(f"{'arm':<22}{'windows':>8}{'%desynced':>11}{'%contact':>10}{'sharpness':>10}")
    for r in rows:
        print(f"{r['arm']:<22}{r['windows']:>8}{r['pct_desynced']:>11}{r['pct_with_contact']:>10}{r['causal_sharpness']:>10}")

    # Does the gate throw away the rare events we are trying to oversample?
    def kept(srcs, **kw):
        r = base(srcs)
        if kw:
            r = r.sync_gate(action="forward", motion="speed", max_lag=3, min_corr=0.6)
        return sum(1 for s in r.collect(n=None).selected
                   if s.features["cr"] > 0 and not truth[s.source_id]["desynced"])
    total = kept(sources)
    print(f"\ncontact windows in well-synced clips kept: window gate {kept(sources, g=1)}/{total}, "
          f"clip-level gate {kept(clip_level_sources(sources))}/{total}")

    # Corpus-wide desync: self-calibrated (relative) checks pass it, an absolute lag gate does not.
    allbad, _ = make_corpus(20, desync_frac=1.0, seed=5)
    kept = base(allbad).sync_gate(action="forward", motion="speed", max_lag=3).collect(n=None).selected
    print(f"\nall-desynced corpus: sync_gate keeps {len(kept)} windows (relative/self-calibrated checks keep ~all)")
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
