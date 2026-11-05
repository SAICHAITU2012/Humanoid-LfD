"""Smoke test: build the darts env headless, load the trained checkpoint, measure speed.

Run:  humanoid_challenge/.venv/bin/python smoke_test.py
"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

REPO = '/Users/saichaitu/Desktop/humaniod/G1-Throw-Darts/lab'
CKPT = REPO + '/experiment2/models/best_verified_20261031'
for p in (REPO, REPO + '/experiment2'):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
torch.set_num_threads(1)
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import runtime_bootstrap  # their win32-only import guard; harmless on macOS
from env import build_env

config = json.load(open(CKPT + '/config.json'))
profile = config['throw_profile']
print('variant:', config['variant'], '| profile:', {k: round(v, 4) if isinstance(v, float) else v for k, v in profile.items()})

t0 = time.perf_counter()
env = build_env(variant='arc_v7', throw_profile=profile, randomize=False, record=False)
t1 = time.perf_counter()
print(f'env built in {t1 - t0:.1f}s; obs={env.observation_space.shape}, act={env.action_space.shape}')

# --- raw physics speed: 150 zero-action control steps (each = 40 x 0.5 ms substeps) ---
obs, _ = env.reset(seed=0)
n = 150
t0 = time.perf_counter()
for i in range(n):
    obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape[0], dtype=np.float32))
dt = time.perf_counter() - t0
print(f'{n} zero-action steps in {dt:.1f}s -> {n / dt:.1f} control steps/s (outcome={info.get("outcome")})')

# --- load the trained policy ---
t0 = time.perf_counter()
model = PPO.load(CKPT + '/policy.zip', device='cpu')
print(f'policy loaded in {time.perf_counter() - t0:.1f}s: {type(model.policy).__name__}')

# --- 2 deterministic episodes through the paired normalizer ---
vec = DummyVecEnv([lambda: env])
norm = VecNormalize.load(CKPT + '/normalize.pkl', vec)
norm.training = False
norm.norm_reward = False
for ep in range(2):
    obs = vec.reset()
    done, steps = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec.step(action)
        done = dones[0]
        steps += 1
    inf = infos[0]
    print(f'ep{ep}: outcome={inf.get("outcome")} bullseye={inf.get("is_success")} '
          f'err_m={inf.get("impact_error_m")} steps={steps}')
print('SMOKE OK')
