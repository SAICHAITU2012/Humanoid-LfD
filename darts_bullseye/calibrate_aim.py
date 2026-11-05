"""Re-center the throw for bullseye precision by searching the calibrated swing
profile (no RL). Coordinate descent over 6 swing parameters, scoring the median
miss distance over 24 fixed domain-randomized seeds with the deterministic policy.

Run (NOT while a training chunk is running -- keep the laptop cool):
  .venv/bin/python calibrate_aim.py --checkpoint runs/finetune_XXXX/policy.zip --rounds 2

Writes runs/calibration_<ts>/best_profile.json and a comparison table.
"""
import os, sys, json, time, argparse, copy
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

REPO = '/Users/saichaitu/Desktop/humaniod/G1-Throw-Darts/lab'
for p in (REPO, REPO + '/experiment2'):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
torch.set_num_threads(1)
import numpy as np
import cloudpickle
from stable_baselines3 import PPO

import runtime_bootstrap
from env import build_env

# (name, base delta round1) -- all validated by CalibratedThrowEnv.configure
PARAMS = [
    ('aim_yaw',        .04),
    ('elbow_back',     .05),
    ('shoulder_finish', .06),
    ('elbow_finish',   .08),
    ('roll_finish',    .08),
    ('rate_scale',     .06),
]

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--episodes', type=int, default=24)
ap.add_argument('--seed-start', type=int, default=91001)
ap.add_argument('--rounds', type=int, default=2)
ap.add_argument('--out-root', default='runs')
ap.add_argument('--objective', choices=('median', 'bull'), default='median',
                help="'median': minimize median miss. 'bull': maximize sub-cm accuracy "
                     "(mean exp(-(err/8mm)^2)) with a board-hit-rate gate.")
ap.add_argument('--no-randomize', action='store_true',
                help='calibrate on nominal conditions (deterministic throw); uses 3 episodes')
ap.add_argument('--profile-json', default=None, help='starting profile JSON (default: checkpoint config)')
args = ap.parse_args()
if args.no_randomize:
    args.episodes = 3  # all resets are identical without randomization

ckpt = os.path.abspath(args.checkpoint)
config = json.load(open(os.path.join(ckpt, 'config.json')))
profile = copy.deepcopy(config.get('throw_profile'))
if profile is not None and 'profile' in profile:
    profile = profile['profile']
if args.profile_json is not None:
    profile = json.load(open(args.profile_json))
    print('starting profile override:', args.profile_json, flush=True)
profile['flight_autopilot'] = True
profile['target_reward_weight'] = 0.  # reward not used here; we score geometry directly

policy = PPO.load(os.path.join(ckpt, 'policy.zip'), device='cpu')
with open(os.path.join(ckpt, 'normalize.pkl'), 'rb') as fh:
    norm = cloudpickle.load(fh)


def evaluate(prof, label=''):
    try:
        env = build_env(variant='arc_v7', throw_profile=prof, randomize=not args.no_randomize, record=False)
    except (ValueError, AssertionError) as e:
        return None  # invalid calibration (joint margins / speed limits)
    errs, outcomes = [], []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action, _ = policy.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
            obs, _, term, trunc, info = env.step(action[0])
            done = term or trunc
        clean = (info.get('release') is not None
                 and info['outcome'] in ('board', 'environment_hit', 'timeout'))
        crossing = info.get('board_plane_crossing')
        board = info.get('outcome') == 'board' and info.get('reward_terms', {}).get('board_impact', 0) > 0
        err = info.get('impact_error_m') if board else (crossing['error_m'] if clean and crossing else None)
        if clean and err is not None:
            errs.append(err)
        outcomes.append(info.get('outcome'))
        board_hits = sum(o == 'board' for o in outcomes)
    env.close()
    if len(errs) < max(1, int(.6 * args.episodes)):  # too many invalid throws
        return None
    if args.no_randomize and len(errs) < args.episodes:
        return None
    res = {'label': label, 'median_err': float(np.median(errs)), 'mean_err': float(np.mean(errs)),
           'min_err': float(np.min(errs)), 'board_hits': board_hits, 'clean': len(errs),
           'episodes': args.episodes}
    if args.objective == 'bull':
        # sub-cm accuracy: throws within ~1 cm dominate; requires most throws to
        # at least reach the board (gate), so one lucky throw cannot win.
        if args.no_randomize:
            res['bull_score'] = float(np.exp(-(np.mean(errs) / .008) ** 2))
        else:
            gate = board_hits >= int(.6 * args.episodes)
            res['bull_score'] = float(np.mean([np.exp(-(e / .008) ** 2) for e in errs])) if gate else -1.
            res['gate_ok'] = bool(gate)
    return res


def score_key(res):
    if args.objective == 'bull':
        return (round(res['bull_score'], 6), -res['median_err'])
    return (res['median_err'], res['mean_err'])


best = evaluate(profile, 'current')
print(f"current profile: median {best['median_err']*100:.2f} cm | mean {best['mean_err']*100:.1f} cm "
      f"| board {best['board_hits']}/{best['episodes']}"
      + (f" | bull_score {best.get('bull_score', 0):.4f}" if args.objective == 'bull' else ''), flush=True)
best_score = score_key(best)
results = [best]
current_best_profile = copy.deepcopy(profile)

t0 = time.perf_counter()
for round_idx in range(1, args.rounds + 1):
    scale = 0.5 ** (round_idx - 1)  # shrinking steps: full delta, then half, quarter, ...
    for name, delta in PARAMS:
        for direction in (+1, -1):
            # coordinate move from the CURRENT best profile
            candidate = copy.deepcopy(current_best_profile)
            candidate[name] = current_best_profile.get(name, 0.) + direction * delta * scale
            res = evaluate(candidate, f'{name}={candidate[name]:+.4f}')
            if res is None:
                print(f"  r{round_idx} {name} {direction * delta * scale:+.3f}: invalid calibration, skip", flush=True)
                continue
            results.append(res)
            improved = score_key(res) < best_score if args.objective == 'median' else score_key(res) > best_score
            msg = f"  r{round_idx} {name} {candidate[name]:+.4f}: median {res['median_err']*100:.2f} cm " \
                  f"| board {res['board_hits']}/{res['episodes']}"
            if args.objective == 'bull':
                msg += f" | bull {res.get('bull_score', -1):.4f}"
            msg += "  <-- best" if improved else ""
            print(msg, flush=True)
            if improved:
                best_score = score_key(res)
                current_best_profile = candidate

out_root = os.path.abspath(args.out_root)
out_dir = os.path.join(out_root, 'calibration_' + time.strftime('%Y%m%d_%H%M%S'))
os.makedirs(out_dir, exist_ok=True)
best_res = min(results, key=score_key) if args.objective == 'median' else max(results, key=score_key)
json.dump({'checkpoint': ckpt, 'episodes': args.episodes, 'seed_start': args.seed_start,
           'objective': args.objective, 'no_randomize': args.no_randomize,
           'results': results, 'best_profile': current_best_profile},
          open(os.path.join(out_dir, 'calibration.json'), 'w'), indent=2)
json.dump(current_best_profile, open(os.path.join(out_dir, 'best_profile.json'), 'w'), indent=2)
print(f"\nbest: median {best_res['median_err']*100:.2f} cm | board {best_res['board_hits']}/{best_res['episodes']}"
      + (f" | bull_score {best_res.get('bull_score', -1):.4f}" if args.objective == 'bull' else '')
      + f" -> wrote {out_dir}/best_profile.json ({time.perf_counter()-t0:.0f}s total)")
