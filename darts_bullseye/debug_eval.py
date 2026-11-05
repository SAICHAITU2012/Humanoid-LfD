"""Debug: why does the policy end episodes in 3 steps? Mirror replay_board_hit.py exactly."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')

REPO = '/Users/saichaitu/Desktop/humaniod/G1-Throw-Darts/lab'
CKPT = REPO + '/experiment2/models/best_verified_20261031'
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

config = json.load(open(CKPT + '/config.json'))
profile = config['throw_profile']

policy = PPO.load(CKPT + '/policy.zip', device='cpu')
with open(CKPT + '/normalize.pkl', 'rb') as fh:
    norm = cloudpickle.load(fh)
print('norm stats: mean[:4]=', norm.obs_rms.mean[:4], 'var[:4]=', norm.obs_rms.var[:4])

env = build_env(variant='arc_v7', throw_profile=profile, randomize=False, record=False)
obs, _ = env.reset(seed=1)
raw0 = obs.copy()
normed0 = norm.normalize_obs(raw0[None, :])[0]
print('raw obs[:6]    =', raw0[:6])
print('normed obs[:6] =', normed0[:6])

for step in range(200):
    action, _ = policy.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
    action = action[0]
    obs, r, term, trunc, info = env.step(action)
    if step < 6 or step % 10 == 0:
        print(f'step {step:3d} t={env.t:.2f} rel_cmd={action[-1]:+.2f} |a|={np.abs(action[:-1]).max():.3f} '
              f'dart={np.round(env.d.xipos[env.sim.dart], 3)} r={r:+.3f}')
    if term or trunc:
        print('DONE:', info.get('outcome'), '| success', info.get('is_success'),
              '| err', info.get('impact_error_m'), '| release', info.get('release'))
        break
