"""Held-out evaluation of a dart-throwing checkpoint (fresh env, deterministic policy).

Mirrors the repo's PhysicsEvaluation protocol: fresh env, seeded resets, domain
randomization on, deterministic actions through the paired normalizer.

Run:
  .venv/bin/python evaluate_bullseye.py --checkpoint <dir-with-policy.zip> \
      --episodes 40 --seed-start 91001 --out runs/baseline.json
"""
import os, sys, json, time, argparse
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

BULL_M = 0.00635          # inner-bull radius
OUTER_BULL_M = 0.0159     # outer-bull radius


def summarize(rows):
    valid = [r for r in rows if r['clean']]
    errors = [r['error_m'] for r in valid]
    scored = [r['error_m'] for r in rows if r['error_m'] is not None]
    return {
        'episodes': len(rows),
        'clean_flight_rate': round(len(valid) / max(1, len(rows)), 4),
        'board_hits': sum(r['board'] for r in rows),
        'bullseyes_inner': sum(e <= BULL_M for e in scored),
        'bullseyes_outer': sum(BULL_M < e <= OUTER_BULL_M for e in scored),
        'within_2cm': sum(e <= 0.02 for e in scored),
        'within_5cm': sum(e <= 0.05 for e in scored),
        'within_10cm': sum(e <= 0.10 for e in scored),
        'median_error_m': round(float(np.median(errors)), 4) if errors else None,
        'mean_error_m': round(float(np.mean(errors)), 4) if errors else None,
        'min_error_m': round(float(np.min(errors)), 4) if errors else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=REPO + '/experiment2/models/best_verified_20261031')
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--seed-start', type=int, default=91001, help='held-out seeds; repo used 76101/83101')
    ap.add_argument('--no-randomize', action='store_true')
    ap.add_argument('--out', default='runs/eval.json')
    args = ap.parse_args()

    ckpt = os.path.abspath(args.checkpoint)
    config = json.load(open(os.path.join(ckpt, 'config.json')))
    profile = config.get('throw_profile')
    if profile is not None and 'profile' in profile:
        profile = profile['profile']

    policy = PPO.load(os.path.join(ckpt, 'policy.zip'), device='cpu')
    with open(os.path.join(ckpt, 'normalize.pkl'), 'rb') as fh:
        norm = cloudpickle.load(fh)

    rows = []
    t0 = time.perf_counter()
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        env = build_env(variant=config.get('variant') or 'arc_v7', throw_profile=profile,
                        randomize=not args.no_randomize, record=False)
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            action, _ = policy.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
            obs, _, term, trunc, info = env.step(action[0])
            done = term or trunc
        board = info.get('outcome') == 'board' and info.get('reward_terms', {}).get('board_impact', 0) > 0
        clean = (info.get('release') is not None
                 and info['outcome'] in ('board', 'environment_hit', 'timeout'))
        crossing = info.get('board_plane_crossing')
        error = info.get('impact_error_m') if board else (crossing['error_m'] if clean and crossing else None)
        rows.append({
            'seed': seed, 'outcome': info.get('outcome'), 'board': bool(board),
            'bullseye': bool(info.get('is_success')),
            'clean': bool(clean and error is not None),
            'error_m': round(float(error), 4) if error is not None else None,
            'mass_scale': info.get('mass_scale'),
        })
        env.close()
        done_str = f"{rows[-1]['outcome']:<16} err={rows[-1]['error_m']}"
        print(f'seed {seed}: {done_str}  ({time.perf_counter() - t0:.0f}s elapsed)', flush=True)

    summary = summarize(rows)
    print('\n=== SUMMARY ===')
    for k, v in summary.items():
        print(f'  {k}: {v}')

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'checkpoint': ckpt, 'config': {'variant': config.get('variant'), 'episodes': args.episodes,
               'seed_start': args.seed_start, 'randomize': not args.no_randomize},
               'summary': summary, 'episodes': rows}, open(out, 'w'), indent=2)
    print('wrote', out)


if __name__ == '__main__':
    main()
