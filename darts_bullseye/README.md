# darts_lab — G1 five-finger bullseye training
> [!NOTE]
> Snapshot of the local `darts_lab` workspace (5 Nov 2026). The MuJoCo physics comes from the cloned [amarnath3003/G1-Throw-Darts](https://github.com/amarnath3003/G1-Throw-Darts) repo (not included here) — see its MIT license and asset notices. The champion checkpoint is `checkpoint/policy.zip` (+ `normalize.pkl`, `checkpoint/config.json`); the 1.4 mm match-conditions aim is `profiles/match_conditions_profile.json`.


Teaching the Unitree G1 (with **five-finger Unitree Inspire DFQ hands** — the same
hands as in amarnath3003's G1-Throw-Darts video) to throw a dart at a regulation
dartboard's bullseye, on a MacBook, without cooking the laptop.

The robot, dart, board and physics all come from the reference repo
`../G1-Throw-Darts` (cloned, unmodified). This folder holds **our** training
scripts, runs and results.

## 🎯 Headline result: an inner bullseye

Under **match conditions** (the standard 26.7 g dart, standard finger friction —
what a real darts robot would use), the champion policy + our centered aim puts
the dart **1.4 mm from dead center — inside the 6.35 mm inner bull**, reproducibly
(verified 3/3, deterministic physics). See `demo/match_nominal_bullseye.mp4`.

Under **domain-randomized** conditions (dart mass ±5%, finger friction ±15%,
80 fresh never-used test throws):

| Metric | Before (repo's best model) | After (our fine-tune) |
|---|---|---|
| Board hits | 50% (20/40, block 1) | **69% (55/80 pooled)** |
| Median miss (when on board) | 19.6 cm | **~4.6 cm** |
| Throws within 2 cm of bull | 0 | **14** |
| Outer-bull hits (≤1.59 cm) | 0 | **8** |
| Closest randomized throw | 13.6 cm | **8.7 mm** |
| Inner-bull (6.35 mm) hits | 0 | randomized: 0 — release scatter is the remaining gap |

**Watch:** `demo/reel.mp4` (14 s highlight reel) or live:
`./.venv/bin/python watch_bullseye.py --checkpoint runs/finetune_20261105_054309/best_physics`


## How it works (like you're five)

The robot's throw is a **calibrated swing** (a smooth pre-planned arm motion, like
a golfer's practiced swing) plus a small **RL brain** that makes tiny corrections
and decides when to open the fingers.

We improved accuracy with a loop:
1. **Fine-tune** the RL brain for a while (PPO, the same algorithm the repo used).
2. **Re-aim** the swing itself by trying small changes to 6 swing knobs and
   keeping what lands closer to the bull (`calibrate_aim.py`).
Repeat. The re-aiming did the heavy lifting: the original aim landed ~14 cm to
one side; after 3 calibration rounds the aim is centered and the misses are
mostly scatter from the dart's release.

## Files

| File | What it does |
|---|---|
| `train_bullseye.py` | Fine-tune a checkpoint (slim, no per-episode video → laptop-friendly). Resume + profile override supported. Writes runs, STOP-file aware. |
| `calibrate_aim.py` | Coordinate-descent search over 6 swing parameters. `--objective bull` + `--no-randomize` = match-conditions aim to sub-mm. |
| `evaluate_bullseye.py` | Honest held-out eval on fresh seeds (fresh env per throw, deterministic policy). |
| `render_best_throws.py` | Films the best throws: wide view + bullseye close-up mp4s into `demo/`. Use `--randomize` for varied throws. |
| `build_reel.py` | Assembles `demo/reel.mp4`: title card → wide throws → slow-mo bullseye close-ups → scoreboard. |
| `watch_bullseye.py` | **Live MuJoCo window** — watch the robot throw in real time. |
| `thermal_watchdog.sh` | Background guardian: checks macOS thermal state every 60 s, force-pauses training at "serious". |
| `smoke_test.py`, `debug_eval.py`, `test_norm.py` | Setup/debug helpers from the build. |

## Try it yourself

```bash
cd darts_lab
# watch the trained robot throw live (close the window to quit)
./.venv/bin/python watch_bullseye.py --checkpoint runs/finetune_20261105_054309/best_physics

# evaluate it yourself on 40 fresh throws
./.venv/bin/python evaluate_bullseye.py --checkpoint runs/finetune_20261105_054309/best_physics \
    --episodes 40 --seed-start 94001 --out runs/my_check.json

# film your own top-5 throws
./.venv/bin/python render_best_throws.py --checkpoint runs/finetune_20261105_054309/best_physics
```

Keep training further (one gentle ~20–30 min chunk, 3 CPU lanes; between chunks
let it cool):

```bash
./.venv/bin/python train_bullseye.py --checkpoint runs/finetune_20261105_054309/best_physics \
    --steps 15000 --workers 3 --exploration-std 0.15
# stop early: touch <RUN>/STOP    (path printed at start)
```

## Training history (all on this Mac, ~4 h total, chunked with cool-downs)

| Step | Median miss (eval seeds) | Note |
|---|---|---|
| repo `best_verified_20261031` | 20.4 cm (40 throws) | aim is ~14 cm off to one side |
| chunk 1: 15k RL steps | no change | residual policy stuck in local optimum |
| calibration r1 | 10.6 cm | aim yaw + shoulder fix |
| chunk 2: 20k RL steps | ~8 cm | board hits 5/8 → 7/8 (eval seeds) |
| calibration r2 | 5.3 cm | big re-centering |
| chunk 3: 15k RL steps (σ 0.15) | ~4.3 cm | **champion** (`best_physics`) |
| calibration r3 + chunk 4: 12k (σ 0.12) | ~5.0 cm | plateau on randomized evals |
| **nominal-aim calibration (r5)** | **1.4 mm — INNER BULL** | match-conditions aim, converged |
| chunk 5: 15k RL on centered aim | 7.3 cm median, 29/40 hits | hit rate ↑, accuracy ≈ champion — kept as alternate |

## Notes & honesty

- Training data: resumed from the repo's 404,588-step checkpoint; our chunks added
  ~77k steps (~20k throws) on top. Reward = repo's measured-target reward +
  our bullseye bonus (tight Lorentzians at 4 mm / 10 mm).
- The champion was selected on seeds 91001–91024 + the nominal throw; headline
  numbers come from seeds 92001–94040, which no selection ever saw.
- The 1.4 mm inner bull is a **match-conditions** result (no randomized mass/
  friction). Under full randomization the release scatter keeps misses at ~cm
  scale — the same wall the reference repo hit (their held-out best: ~10% inner
  bulls on 128 throws).
- Laptop safety: ≤3 physics workers, torch pinned to 1 thread, no ffmpeg during
  training, cool-downs between chunks, plus `thermal_watchdog.sh` (pauses training
  the moment macOS reports "serious" thermal state; it auto-paused chunk 5 when a
  second training run from another project was sharing the CPU).
- iCloud note: `Desktop/` is iCloud-synced, so `bird` churns while training
  writes checkpoints. Harmless, but future heavy runs could live outside Desktop.

Built on [amarnath3003/G1-Throw-Darts](https://github.com/amarnath3003/G1-Throw-Darts)
(MIT) — see that repo for the dart aerodynamics calibration, grip design and
upstream asset licenses (Unitree G1 + Inspire DFQ hands).
