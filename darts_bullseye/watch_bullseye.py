"""Watch the trained five-finger G1 throw darts in a live MuJoCo window.

Real-time pacing (one 20 ms control step per ~20 ms), auto-reset after each
throw, outcome + miss distance printed in the terminal.

Run:
  .venv/bin/python watch_bullseye.py --checkpoint runs/finetune_XXXX/policy.zip
  (or point at the repo's original: ../G1-Throw-Darts/lab/experiment2/models/best_verified_20261031/policy.zip)

Close the window (or Ctrl-C in the terminal) to quit.
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')

REPO = '/Users/saichaitu/Desktop/humaniod/G1-Throw-Darts/lab'
for p in (REPO, REPO + '/experiment2'):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
torch.set_num_threads(1)
import numpy as np
import cloudpickle
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

import runtime_bootstrap
from env import build_env

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True, help='directory containing policy.zip (+normalize.pkl)')
ap.add_argument('--randomize', action='store_true', help='domain-randomized throws (default: clean)')
ap.add_argument('--pause', type=float, default=1.2, help='seconds between throws')
args = ap.parse_args()

ckpt = os.path.abspath(args.checkpoint)
config = json.load(open(os.path.join(ckpt, 'config.json')))
profile = config.get('throw_profile')
if profile is not None and 'profile' in profile:
    profile = profile['profile']

policy = PPO.load(os.path.join(ckpt, 'policy.zip'), device='cpu')
with open(os.path.join(ckpt, 'normalize.pkl'), 'rb') as fh:
    norm = cloudpickle.load(fh)

env = build_env(variant='arc_v7', throw_profile=profile, randomize=args.randomize, record=False)
sim = env.sim

print('starting viewer... (close the window or Ctrl-C to quit)')
obs, _ = env.reset(seed=0)
throws = {'board': 0, 'total': 0, 'errors': []}
with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
    # start looking at the robot from the side, board ahead
    viewer.cam.lookat[:] = [1.5, 0.0, 1.4]
    viewer.cam.distance = 4.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -8
    try:
        while viewer.is_running():
            done = False
            step_start = time.perf_counter()
            while viewer.is_running() and not done:
                action, _ = policy.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
                obs, _, term, trunc, info = env.step(action[0])
                done = term or trunc
                viewer.sync()
                # real-time pacing: one control step = 20 ms of sim time
                lag = 0.02 - (time.perf_counter() - step_start)
                if lag > 0:
                    time.sleep(lag)
                step_start = time.perf_counter()
            if not viewer.is_running():
                break
            throws['total'] += 1
            outcome = info.get('outcome')
            err = info.get('impact_error_m')
            if info.get('outcome') == 'board' and info.get('reward_terms', {}).get('board_impact', 0) > 0:
                throws['board'] += 1
                throws['errors'].append(err)
            bull = bool(info.get('is_success'))
            msg = f"throw {throws['total']}: {outcome}"
            if err is not None:
                msg += f" | miss {err * 100:.1f} cm"
            msg += " | BULLSEYE!" if bull else ""
            msg += f" | board rate {throws['board']}/{throws['total']}"
            print(msg, flush=True)
            time.sleep(args.pause)
            if not viewer.is_running():
                break
            obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        if throws['total']:
            mean_err = np.mean(throws['errors']) * 100 if throws['errors'] else float('nan')
            print(f"\nsession: {throws['board']}/{throws['total']} board hits, mean miss "
                  f"{mean_err:.1f} cm (on hits)")
print('viewer closed.')
