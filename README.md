# Humanoid-LfD — Teaching a Unitree G1 from human video

**Participant:** krish · **Challenge:** *Teach a Humanoid With Your Own Hands* (Awign × Armature AI Labs) · **Deadline:** 4 Nov 2026

This project teaches the MuJoCo Unitree G1 (Dex3-1 three-finger hand) to reproduce a human cup pick-and-place demonstration, using a personally recorded phone video as the core data source. It is built as a **two-stage learning-from-demonstration study**:

- **Experiment A (public data):** a BC policy trained on 238 real VR-teleoperated G1 Dex3 episodes (MIT-licensed, Hugging Face), plus kinematic replay of those episodes in this repo's own cup scene.
- **Experiment B (own data):** a landmark→joint mapper trained on **5 videos I recorded myself** (cup → saucer, steel cup on a paper plate), producing **7 independent finger-joint targets** — the capability the public dataset's single coupled `squeeze` scalar cannot express.

Everything below is produced by code in this repository and reproducible with the commands in §7.

---

## 1. Pipeline

```
                       ┌────────────────────────────────────────────────────────┐
 my phone video ──► MediaPipe hand landmarks ──► retarget (7 G1 finger joints) ──► MLP mapper ──► MuJoCo G1 rollout
 (Experiment B)     extract_hand.py             retarget_hand.py                  train_policy.py   run_g1_replay.py --model

 public G1 Dex3 dataset ──► BC policy (state → 7 arm joints + squeeze) ──► closed-loop evaluation
 (Experiment A)             train_public_baseline.py / train_chunked_baseline.py
```

Also included: a randomized cup pick-and-place simulation (`src/cup_env.py`) with a scripted expert (**98/100 successful demos**), a BC scaling study {5, 25, 100 demos}, and interactive viewers (`src/view_g1_*.py`).

## 2. Results at a glance

### The A/B comparison (submission centerpiece)

| | **Experiment A — public data** | **Experiment B — my recording** |
|---|---|---|
| Source | 238 VR-teleop episodes, 52,952 frames (MIT) | 5 takes I recorded, 3,651 frames / 2,294 aligned |
| Finger supervision | 1 coupled `grasp_squeeze` scalar | **7 independent finger joints, full range (0→1.35 rad)** |
| Model | BC: 14→128→64→8 MLP | Mapper: 63→128→64→7 MLP, val RMSE **0.452** (best in project) |
| Task execution | closed-loop diverges (0.51–1.79 rad); replay holds cup ~0.5 s then slips | mapper-driven kinematic reproduction, finite 74 s rollout (`outputs/g1_hand_replay_experimentB_all.mp4`) |
| Role | baseline + quantified gap | **satisfies the challenge's hard rule** |

### Honest negative results (measured, not hidden)

1. **BC compounding error, quantified.** Teacher-forced action prediction: **0.085 rad**; closed-loop rollout: **0.51 rad** (chunked ACT-style variant: 0.53; every-frame replanning: 1.79). The cup-task BC policies reach ~0.076 val RMSE yet **0/30 closed-loop successes**. Root cause: a policy trained on static demos never learns to recover from its own mistakes. This is exactly what the requested GPU credits fund (DAgger at volume, §`PROOF_OF_WORK.md`).
2. **Force closure does not transfer through position replay.** The public dataset's real grasp is a tactile-servoed side pinch; kinematic replay reproduces the *motion* but holds the cup only ~0.5 s before it slips (16/66 held-lift frames, best of 32 placement/radius/overshoot configurations).
3. **The scripted expert's knife-edge sensitivity.** Cup spawn ±1.5 cm, goal ±1 cm, friction ±15%, or mass ±20% each collapse expert success to 0–35% — the sensitivity frontier that defines the robust-training compute needed.

### Trained models (all in `models/`, all small, all committed)

| Model | Data | Result |
|---|---|---|
| `public_baseline.pt` | 238 public G1 Dex3 episodes | held-out action RMSE **0.0847 rad** |
| `public_baseline_chunked.pt` | ACT-style chunked variant | closed-loop 0.53 rad (best closed-loop so far) |
| `hand_mapper_experimentB.pt` | **my 5 recordings** | val RMSE **0.452** — best mapper in project |
| `hand_mapper.pt` / `hand_mapper_cup_train*.pt` | stock/validation clips | 0.595 / 2.563 / 2.228 |
| `cup_bc_{5,25,100}.pt` | scripted cup demos | val RMSE ~0.076; closed-loop 0/30 (§ negative results) |

## 3. Data provenance & licenses (declared in full)

| Data | License | Use |
|---|---|---|
| `lewislf/G1_Dex3_pick_white_cup_right8_1squeeze` (Hugging Face) | MIT | Experiment A training + replay. Re-download: `bash scripts/download_public_dataset.sh` (5.1 GB, not committed) |
| `demo3.mp4`, `demo5.mp4` (**my own recordings**, included) | mine | Experiment B + representative samples; hand + tabletop only, no private content |
| Ghana sign-alphabet clip (Wikimedia Commons) | CC | pipeline validation only, declared |
| Two Envato stock screen recordings | *watermarked previews* | pipeline validation only — **not committed, not training data** |
| MuJoCo Menagerie `unitree_g1` | BSD-3-Clause (upstream) | robot model, commit pinned in `vendor/MENAGERIE_COMMIT.txt` |

The WhatsApp challenge export and other participants' media were **never** used as data.

## 4. Key evidence files

- `outputs/g1_hand_replay_experimentB_all.mp4` — 74 s G1 rollout, 7 finger joints driven by the Experiment-B mapper over all 5 of my takes
- `outputs/g1_hand_replay_experimentB_demo3.mp4` — the 100%-tracked calibration take, replayed
- `outputs/fig_public_baseline_predictions.png` — BC vs real teleop on held-out episodes
- `outputs/replay_public_episode.mp4` + `rollout_public_trace.csv` — public-episode kinematic replay (incl. the force-closure slip, honestly traceable)
- `outputs/fig_expert_episode.png`, `fig_bc_scaling.png`, `cup_eval_*.json` — cup-task expert + scaling study
- `PROOF_OF_WORK.md` — full measurement report + the concrete GPU-credit request

## 5. Setup

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash scripts/setup_menagerie.sh          # sparse-clones the Menagerie G1 (BSD-3)
bash scripts/download_hand_model.sh      # MediaPipe hand_landmarker.task
.venv/bin/python src/validate_setup.py
```

## 6. Repo layout

```
src/  extract_hand.py retarget_hand.py train_policy.py run_g1_replay.py     # video → G1 fingers
      train_public_baseline.py train_chunked_baseline.py replay_public_episode.py  # Experiment A
      evaluate_public_baseline.py                                            # figures + rollout
      cup_env.py train_cup_bc.py                                             # randomized cup task + BC study
      run_cup_pick_place.py run_grasp_demo.py run_stand_preview.py           # scripted demos
      view_g1_interactive.py view_g1_web.py g1_control_server.py             # live viewers
scripts/  setup_menagerie.sh download_hand_model.sh download_public_dataset.sh
data/raw/  demo3.mp4 demo5.mp4 (my recordings; full set regenerates everything)
outputs/  videos, traces, figures (evidence subset committed)
```

## 7. Reproduce the headline numbers

```bash
# Experiment A (public data; downloads 5.1 GB first)
bash scripts/download_public_dataset.sh
.venv/bin/python src/train_public_baseline.py          # 0.0847 rad held-out
.venv/bin/python src/replay_public_episode.py          # kinematic replay + force-closure finding

# Experiment B (my recordings; demo3/demo5 committed, full set regenerates)
.venv/bin/python src/extract_hand.py data/raw/demo3.mp4 --out data/processed/landmarks_demo3.jsonl
.venv/bin/python src/retarget_hand.py data/processed/landmarks_demo3.jsonl --out data/processed/actions_demo3.csv
.venv/bin/python src/train_policy.py data/processed/landmarks_experimentB.jsonl \
    data/processed/actions_experimentB.csv --out models/hand_mapper_experimentB.pt   # val RMSE 0.452
.venv/bin/python src/run_g1_replay.py --landmarks data/processed/landmarks_experimentB.jsonl \
    --model models/hand_mapper_experimentB.pt --video outputs/g1_hand_replay_experimentB_all.mp4

# Cup task + scaling study
.venv/bin/python src/train_cup_bc.py generate --demos 100 --out data/processed/cup_bc_demos.npz
.venv/bin/python src/train_cup_bc.py train --subset 100 --out models/cup_bc_100.pt
```

## 8. Roadmap

1. **GPU credits (see `PROOF_OF_WORK.md`):** DAgger at volume to close the 0.085 → 0.51 rad closed-loop gap; temporal mappers; robust-training over the randomization frontier.
2. Composed demo: my recorded finger choreography + scripted arm reach in the calibrated cup scene, with fingertip-contact confirmation.
3. Extending to cup position / appearance variation once robust training works.
