# Sync-aware curation for action-conditioned world models

## The problem

An action-conditioned world model learns "action → what happens next" from (frame, action) pairs.
That only works if the action log and the video are on the same clock. In real gameplay corpora
they often aren't: capture latency, dropped frames and variable frame rate shift the action log
by several ticks in a subset of clips. The model then learns a smeared or wrong causal kernel
("W" seems to cause motion 0.8 s later) and the damage is silent. Nothing looks wrong in any
single frame, and it can't be fixed by collecting more data.

The SDK's `action_state_consistency` addresses part of this, but it judges isolated key presses
against a baseline. It can't say *how far* a clip is shifted, and its self-calibrated mode
"structurally cannot find a corpus-wide problem" (its own docs). It also cannot be reduced to a
clip verdict in a live pipeline (documented gap). I wanted a gate that is **absolute** (latency
in ticks, not relative to the corpus), **clip-aware**, and **composes with the rare-event
oversampling** the SDK already provides, since the rare moments (wall hits) are exactly the ones
a naive gate tends to throw away.

## Approach (all in `src/`, runs with `python src/main.py`)

| Piece | SDK surface used |
|---|---|
| `desync/action-motion-lag` – per-window cross-correlation of an action channel against a measured-motion channel → `lag_ticks`, `lag_corr` | `@register_batch_transform` (`ctx.input_token`, `valid_steps`) |
| `desync/sync-gate` – transform + `.filter(lag_corr >= c)` + `.filter(lag_ticks <= L)`; unmeasurable windows are rejected (fail closed) | `@register_operator` → shows up as `pipeline.sync_gate(...)` |
| Stock selection after the gate | `rare_event_coverage` from `kuration.selection` |
| Corpus, schema, ticks | `input(dict_of_ticks, schema=SchemaSpec)`, `.window(64, stride=32)`, `.measure`, `.explain()`, `.output(...)` (shards + `config.json`) |
| Clip-level gate: pass 1 estimates window lags, takes each clip's median over confident windows, pass 2 re-runs on admitted clips | `collect(n=None)` + plain Python |

In production the two channels come from the SDK's own operators (`pixel_shift` or
`slam_reconstruct` for measured motion; `collision` for the contact flag). Here they are
precomputed per-tick columns, as in `examples/06`, because I had no video, GPU or CS2 meshes.

**Data.** 60 synthetic clips × 600 ticks. 25% are desynced (action→motion latency of 6–10 ticks
instead of 1–2). About 1% of ticks are wall hits, where speed collapses. Ground truth is held out
and used for scoring only.

## Result (same corpus, same 120-window budget, same seed)

| arm | % windows from desynced clips | % windows with a wall hit | causal sharpness* |
|---|---|---|---|
| naive random draw | 21.7 | 13.3 | 0.661 |
| stock `rare_event_coverage` | 19.2 | 100 | 0.696 |
| + `sync_gate` per window | **0.0** | 77.5 | 0.838 |
| + `sync_gate` per clip | **0.0** | **100** | **0.849** |

\*Downstream proxy: regress speed_t on action_{t-k} (k = 0..12) over the selected windows and
measure how much of the learned kernel sits on the true 1–2 tick latency. It is a stand-in for
"how crisp an action→effect relationship will a model see", **not** a trained world model.

Three findings:

1. **Stock rare-event oversampling does nothing about contamination.** `rare_event_coverage` hit
   100% wall-hit windows, but 19% of the budget still came from desynced clips and sharpness moved
   only 0.661 → 0.696. Oversampling is only as good as the pool it draws from.
2. **A per-window gate throws away the events you are oversampling.** It kept only 93 of 160
   wall-hit windows from *healthy* clips. In a diagnostic run, 48 of the 65 dropped ones were windows
   where W was held throughout, so there is no onset to correlate. Aggregating to clip level keeps 160/160.
3. **Absolute gating handles the corpus-wide failure.** On a corpus where every clip is desynced
   the gate keeps 0 windows; a relative (self-calibrated) check would pass all of them.

## SDK gaps I hit (and what I did)

1. **`.output()` crashes on Windows.** `io._fsync_file` opens the file `O_RDONLY` and calls
   `os.fsync`, which raises `EBADF` there. I patched it inside `main.py` (and did not edit the SDK).
   The fix is `O_RDWR`.
2. **No clip-level reduction in a live pipeline.** Metrics are per window and there is no
   group-by-source reduction, nor a way to broadcast a clip statistic back to its windows. I needed
   a two-pass workaround. This is the single biggest feature gap: `action_state_consistency`,
   `slam_quality` and my gate all want a clip-level verdict.
3. **No plain uniform-random sampler.** `.sample("equal")` requires a bin axis, and `collect()`
   requires a count, so even a naive baseline needs a workaround.
4. **Feature-name collisions.** `rare_event_coverage` hard-codes its measure name, so the same
   name can't be pre-measured for evaluation. A minor point, but it makes composition fiddly.
5. **Design tension I could not test** (it needs video and meshes): `action_state_consistency`
   treats "W pressed, camera didn't move" as inconsistent, while `collision` wants exactly those
   windows oversampled. Used together naively they should cancel out. The gate needs a way to
   take a contact mask from `collision`. My experiment's analogue is finding 2.
6. **Not exercised:** `pixel_shift`, `collision`, `slam_*` and `join_video` on real media. Nothing
   here shows how robust the lag estimator is to real optical-flow noise.

## Limits of this evidence

Synthetic data with a clean linear response; the proxy metric is mine; one seed. The numbers show
the mechanism and the composition, not an expected gain on CS:GO. The next step would be the same
gate on the TeaPearce or Open P2P adapters, using `pixel_shift` for the motion channel, and
checking a trained DIAMOND-style model's controllability against the proxy.
