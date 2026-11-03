# Humanoid-LfD — Teaching a Unitree G1 with your own hands

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Sim: MuJoCo 3.17](https://img.shields.io/badge/sim-MuJoCo%203.17-green.svg)](https://mujoco.org/)

> **Challenge:** *Teach a Humanoid With Your Own Hands* (Awign × Armature AI Labs) · **Participant:** krish · **Deadline:** 4 Nov 2026
> Built on an Apple-Silicon MacBook (MPS, no NVIDIA GPU) — every number below is measured by code in this repo.

A learning-from-demonstration (LfD) study on the MuJoCo Unitree G1 (Dex3-1 hand): a **personally recorded phone video** is turned into finger-joint choreography on the robot, and it is benchmarked against a behavior-cloning (BC) policy trained on **238 real VR-teleoperated G1 episodes**. The comparison between the two data sources is the submission's centerpiece — and every negative result is measured, not hidden.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Results & Benchmarks](#-results--benchmarks)
- [Journey: what worked, what didn't, assumptions](#-journey-what-worked-what-didnt-assumptions)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Appendix](#-appendix)

---

## 📖 Overview

The challenge's hard rule is that the robot must be taught from **your own** demonstration. This project answers it with a controlled A/B study:

- **Experiment B (own data — the hard rule):** 5 videos I recorded myself (steel cup → paper-plate saucer, right hand, fixed 7-phase protocol). MediaPipe tracks the hand; a retargeting step maps landmarks to **7 independent G1 finger joints with full-range closures** (index/middle 0→1.35 rad, thumb to −0.84 rad); an MLP mapper learns landmark→joint; the trained mapper drives a G1 rollout in MuJoCo. **Val RMSE 0.452 — the best mapper in the project**, directly showing the data-scaling effect of 5 consistent takes vs a single clip.
- **Experiment A (public-data baseline):** BC policy trained on 238 MIT-licensed VR-teleoperated G1+Dex3 cup-picking episodes (52,952 frames @ 30 Hz). Held-out action RMSE **0.0847 rad** — excellent on paper, and an honest closed-loop failure that quantifies *why* (teacher-forced 0.085 rad vs closed-loop 0.51 rad).

**Why not RL — honestly.** The template's "RL fine-tuning" stage is this project's *measured next step*, not its current method. Both experiments independently reproduce the classic BC compounding-error failure: a policy trained on static demonstrations never sees its own mistakes. A first DAgger round at small volume (30 relabeled rollouts) made the policy worse. Closing the 0.085 → 0.51 rad gap with DAgger/interaction training **at volume** is precisely the GPU-credit request in [`PROOF_OF_WORK.md`](PROOF_OF_WORK.md).

## ✨ Key Features

* **End-to-end video → G1 fingers pipeline:** phone video → MediaPipe landmarks → retargeted finger targets → trained MLP mapper → MuJoCo G1 rollout (finite, no drift, 74 s over all takes).
* **Two-source A/B design:** public 238-episode teleop baseline vs my own 5-take recording, same scene, same evaluation discipline.
* **Randomized cup pick-and-place environment** (`src/cup_env.py`) with contact-based success checks and a scripted expert producing **98/100 successful demos**.
* **BC scaling study** {5, 25, 100 demos} with held-out validation and a measured negative result (closed-loop 0/30).
* **Composed demo:** finger choreography from my video drives the hand while scripted IK arm stages run reach → grasp → lift on the calibrated cup scene.
* **Interactive control:** browser-based live viewer (14 arm/hand sliders, Walk/Halt/Reset) and a **quasi-static walking gait** tuned headlessly (~150 sim runs) — walks **+0.55 m in ~4 s**, then stumbles, honestly disclosed.
* **Fully seeded & reproducible:** every headline number regenerates from the commands in this README.

## 🏗️ System Architecture

```
                       ┌────────────────────────────────────────────────────────────┐
 my phone video ──► MediaPipe hand landmarks ──► retarget (7 G1 finger joints) ──► MLP mapper ──► MuJoCo G1 rollout
 (Experiment B)     extract_hand.py             retarget_hand.py                  train_policy.py   run_g1_replay.py

 public G1 Dex3 dataset ──► BC policy (state → 7 arm joints + squeeze) ──► closed-loop evaluation
 (Experiment A)             train_public_baseline.py / train_chunked_baseline.py

 scripted expert ──► 11,788 demos ──► BC policies ──► closed-loop eval (scaling study)
                            cup_env.py + train_cup_bc.py
```

Three learning stages:
1. **Perception & retargeting:** MediaPipe hand landmarks (21 per frame) are retargeted to the G1 Dex3 hand's 7 right-finger joints, with per-finger range calibration from the video's own grasp geometry.
2. **Learning:** small MLPs (63→128→64→7 mapper; 14→128→64→8 BC policy) trained with seeded, validated splits. Temporal/chunked variants included (ACT-style execute-8).
3. **Execution & measurement:** kinematic replay, closed-loop rollouts, and contact-based success metrics in MuJoCo — the same criterion for expert, BC, and replay.

## 📁 Repository Structure

```text
├── src/                        # all pipeline code
│   ├── extract_hand.py         #   video → MediaPipe landmarks
│   ├── retarget_hand.py        #   landmarks → 7 G1 finger joints
│   ├── train_policy.py         #   landmark→joint mapper training
│   ├── run_g1_replay.py        #   mapper-driven G1 rollout (+video)
│   ├── train_public_baseline.py / train_chunked_baseline.py   # Experiment A BC
│   ├── replay_public_episode.py / evaluate_public_baseline.py # replay + figures
│   ├── cup_env.py / train_cup_bc.py      # randomized cup task, expert, BC study
│   ├── composed_demo.py        #   my-video fingers + IK arm composed task
│   ├── gait.py / tune_walk.py / walk_check.py  # quasi-static walking gait
│   ├── view_g1_web.py / g1_control_server.py   # live browser viewer
│   └── run_cup_pick_place.py / run_grasp_demo.py / run_stand_preview.py
├── scripts/                    # setup_menagerie.sh, download_hand_model.sh, download_public_dataset.sh
├── models/                     # all trained weights + training histories (committed, small)
├── outputs/                    # videos, traces, figures (evidence subset committed)
├── data/                       # raw demos (mine) + processed arrays (regenerated)
├── vendor/MENAGERIE_COMMIT.txt # pinned MuJoCo Menagerie unitree_g1 (BSD-3)
├── PROOF_OF_WORK.md            # full measurement report + GPU-credit request
├── requirements.txt
└── README.md
```

## 💻 Installation

Tested on Apple Silicon (macOS, Python 3.12) — no NVIDIA GPU needed for anything here.

```bash
git clone https://github.com/SAICHAITU2012/Humanoid-LfD.git
cd Humanoid-LfD

python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

bash scripts/setup_menagerie.sh          # sparse-clones the Menagerie G1 (BSD-3)
bash scripts/download_hand_model.sh      # MediaPipe hand_landmarker.task
.venv/bin/python src/validate_setup.py   # sanity check
```

## 📊 Dataset Preparation

| Data | License | Use |
|---|---|---|
| `lewislf/G1_Dex3_pick_white_cup_right8_1squeeze` (Hugging Face) — 238 VR-teleop episodes, 52,952 frames @30 Hz, head cam + depth + 108-sensor tactile | MIT | Experiment A. Download: `bash scripts/download_public_dataset.sh` (5.1 GB, not committed) |
| `demo3.mp4`, `demo5.mp4` — **my own recordings** (cup → saucer), included in `data/raw/` | mine | Experiment B; the full 5-take set regenerates everything |
| Ghana sign-alphabet clip (Wikimedia Commons), 1441 tracked frames | CC | pipeline validation only, declared |
| Two Envato stock screen recordings (watermarked) | preview only | pipeline validation only — **not committed, not training data** |
| MuJoCo Menagerie `unitree_g1` | BSD-3 | robot model, commit pinned |

The WhatsApp challenge export and other participants' media were **never** used as data.

## 🚀 Usage Guide

### 1. Experiment B — my own video teaches the robot (the challenge's hard rule)
```bash
.venv/bin/python src/extract_hand.py data/raw/demo3.mp4 --out data/processed/landmarks_demo3.jsonl
.venv/bin/python src/retarget_hand.py data/processed/landmarks_demo3.jsonl --out data/processed/actions_demo3.csv
.venv/bin/python src/train_policy.py data/processed/landmarks_experimentB.jsonl \
    data/processed/actions_experimentB.csv --out models/hand_mapper_experimentB.pt   # val RMSE 0.452
.venv/bin/python src/run_g1_replay.py --landmarks data/processed/landmarks_experimentB.jsonl \
    --model models/hand_mapper_experimentB.pt --video outputs/g1_hand_replay_experimentB_all.mp4
```

### 2. Experiment A — public-data BC baseline
```bash
bash scripts/download_public_dataset.sh                 # 5.1 GB
.venv/bin/python src/train_public_baseline.py           # held-out RMSE 0.0847 rad
.venv/bin/python src/replay_public_episode.py           # kinematic replay + force-closure finding
```

### 3. Cup task: expert, BC training, scaling study
```bash
.venv/bin/python src/train_cup_bc.py generate --demos 100 --out data/processed/cup_bc_demos.npz   # 98/100 expert
.venv/bin/python src/train_cup_bc.py train --subset 100 --out models/cup_bc_100.pt
.venv/bin/python src/train_cup_bc.py evaluate --model models/cup_bc_100.pt --attempts 30 --out outputs/cup_eval_bc100.json
```

### 4. Composed demo — my video's fingers + scripted IK arm
```bash
.venv/bin/python src/composed_demo.py --take demo1 --video outputs/composed_task_demo.mp4
```

### 5. Interactive viewer & walking gait (live, mjpython)
```bash
.venv/bin/python src/view_g1_web.py        # MuJoCo window (mjpython)
.venv/bin/python src/g1_control_server.py  # browser page at 127.0.0.1:8765 — 14 sliders, Walk/Halt/Reset
.venv/bin/python src/tune_walk.py          # headless gait sweep (~150 sim runs)
```

## ⚙️ Configuration

No YAML/Hydra layer — every script is CLI-flag driven (fewer moving parts for a from-scratch build). Key flags:

| Script | Flag | Meaning |
|---|---|---|
| `train_policy.py` | `--out`, `--history-out` | mapper weights + per-epoch history JSON |
| `train_cup_bc.py` | `generate --demos N` | expert demonstrations (per-stage ±0.03 s timing jitter) |
| `train_cup_bc.py` | `train --subset N` | dataset-scaling experiment {5, 25, 100} |
| `train_cup_bc.py` | `evaluate --attempts N` | seeded closed-loop success measurement |
| `run_g1_replay.py` | `--model`, `--video` | which mapper drives the G1, output video |
| `composed_demo.py` | `--take`, `--cup-radius` | which of my recordings; cup size from the video's grasp triangle |

## 📈 Results & Benchmarks

### The A/B comparison (submission centerpiece)

| | **Experiment A — public data** | **Experiment B — my recording** |
|---|---|---|
| Source | 238 VR-teleop episodes, 52,952 frames (MIT) | 5 takes I recorded, 3,651 frames / 2,294 aligned |
| Finger supervision | 1 coupled `grasp_squeeze` scalar | **7 independent finger joints, full range (0→1.35 rad)** |
| Model | BC: 14→128→64→8 MLP | Mapper: 63→128→64→7 MLP, val RMSE **0.452** (best in project) |
| Task execution | closed-loop diverges (0.51–1.79 rad); replay holds cup ~0.5 s then slips | mapper-driven kinematic reproduction, finite 74 s rollout |
| Role | baseline + quantified gap | **satisfies the challenge's hard rule** |

### Behavior cloning: the measured failure mode

| Policy | Teacher-forced / val RMSE | Closed-loop success |
|---|---|---|
| Public-data BC (`public_baseline.pt`) | **0.085 rad** | **0.505 rad** divergence on held-out episode |
| Chunked ACT-style (execute-8) | — | 0.53 rad (0.33 first half → 0.73 second half) |
| Chunked, re-plan every frame | — | 1.79 rad (replanning dither) |
| Cup-task BC {5, 25, 100 demos} | ~0.076 | **0/30** (scripted expert: 47%) |
| DAgger round 1 (30 relabeled rollouts) | — | made it worse (0%-success teacher labels off-distribution states) |

### Trained models (all in `models/`, all committed)

| Model | Data | Result |
|---|---|---|
| `hand_mapper_experimentB.pt` | **my 5 recordings** | val RMSE **0.452** — best mapper in project |
| `public_baseline.pt` | 238 public episodes | held-out action RMSE **0.0847 rad** |
| `public_baseline_chunked.pt` | chunked variant | best closed-loop attempt (0.53 rad) |
| `hand_mapper.pt` / `hand_mapper_cup_train*.pt` | stock/validation clips | 0.595 / 2.563 / 2.228 (data-scaling effect visible) |
| `cup_bc_{5,25,100}.pt` | scripted cup demos | val ~0.076; closed-loop 0/30 |

### Walking & composed demo

* **Quasi-static gait:** +0.55 m forward in ~4 s (several genuine steps), then stumbles — scripted, not learned; see [Journey](#-journey-what-worked-what-didnt-assumptions).
* **Composed demo:** my demo1 finger choreography closes the grasp on the cup while the IK arm runs reach → grasp → lift; grasp verified by contact metrics (up to 13 simultaneous hand–cup contacts, tilt < 6°). The carry stage remains limited by the same grasp-force-feedback gap found in Experiment A.

## 🗺️ Journey: what worked, what didn't, assumptions

**What worked**
- The full video→fingers pipeline, end to end, on a laptop: my own 5-take recording produced the project's best mapper (0.452) and a 74 s finite robot rollout — the challenge's core rule satisfied with personal data.
- Controlled A/B design: two independent datasets reaching the same conclusion (BC compounding error) is far stronger evidence than either alone.
- Contact-based success criteria and seeded evaluations — every number in this README is reproducible.
- The walking gait after fixing three measured bugs (knee-phase vs swing direction with a 1:1 FK ankle-leveling rule; a backwards pitch-feedback sign found via a push-recovery test; locked swing knees fixed with a body-dip unloading trick).

**What didn't work**
- Vanilla BC closed-loop, everywhere: 0/30 on the cup task despite ~0.076 val RMSE; 0.51 rad divergence on the public episodes. Teacher-forced accuracy is excellent — the gap is the entire finding.
- DAgger at 30 rollouts (made the policy worse — a 0%-success teacher is a bad labeler; literature says volume fixes this).
- Force closure through position replay: the real grasp is a tactile-servoed side pinch (108 sensors); kinematic replay holds the cup ~0.5 s, then it slips.
- The scripted expert is knife-edge: cup spawn ±1.5 cm, goal ±1 cm, friction ±15%, or mass ±20% each collapse success to 0–35%.

**Assumptions made**
- Kinematic retargeting of MediaPipe landmarks is a valid teaching signal for finger *choreography* (position-level, not force-level).
- Fixed-base simulation is acceptable for tabletop manipulation studies (the standing keyframe is not dynamically stable under arm motion — disclosed).
- A small MLP suffices at this data scale; temporal policies are the next step, not a prerequisite, for the current conclusions.
- Contact-based success (distance + tilt + rest + hand-open) is an honest stand-in for task success.

## 🤝 Contributing

This is a challenge submission, but issues and PRs are welcome: fork → feature branch → commit → PR.

## 📜 License

Distributed under the MIT License — see [LICENSE](LICENSE).

## 🙏 Acknowledgements

* [MuJoCo](https://mujoco.org/) and [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3) for the Unitree G1 model.
* [MediaPipe](https://developers.google.com/mediapipe) for hand landmark tracking.
* The MIT-licensed Hugging Face dataset `lewislf/G1_Dex3_pick_white_cup_right8_1squeeze` for the Experiment-A baseline.
* Awign × Armature AI Labs for the challenge.

---

## 📎 Appendix

### A. Key evidence files
- `outputs/g1_hand_replay_experimentB_all.mp4` — 74 s G1 rollout, 7 finger joints driven by my recordings' mapper
- `outputs/g1_hand_replay_experimentB_demo3.mp4` — the 100%-tracked take replayed
- `outputs/composed_task_demo.mp4` / `outputs/side_by_side_task.mp4` — composed demo + side-by-side submission video
- `outputs/replay_public_episode.mp4` — public-episode replay incl. the force-closure slip
- `outputs/fig_public_baseline_predictions.png`, `fig_bc_scaling.png`, `fig_expert_episode.png`, `cup_eval_*.json`
- [`PROOF_OF_WORK.md`](PROOF_OF_WORK.md) — full measurement report

### B. GPU-credit request (concrete — why compute is the bottleneck)
Local MPS training of these MLPs takes seconds; the bottleneck is **iteration count and rollout volume**:
1. DAgger at volume (thousands of relabeled rollouts/round) — the measured fix for the 0.085 → 0.51 rad gap.
2. Temporal mappers (1D-CNN/LSTM over observation history) and wider/ensembled policies.
3. Robust-training over the randomization frontier already implemented in `cup_env.py`.
4. Dataset-scaling curve extended past 100 demos toward 300+.
5. Closed-loop correction at scale on the real 238-episode distribution — the difference between a *reproducing* hand and an *executing* one.

### C. Honest provenance notes
- Envato stock clips are watermarked pipeline-validation inputs only — never training data, never published as such.
- The cup simulation runs fixed-base; disclosed and standard for tabletop studies.
- The scripted expert is the demonstrator; the learned policies and their measured failure mode are reported as-is.
- Real-robot transfer was out of scope on this hardware/timeline; all claims are simulation-based and reproducible.
