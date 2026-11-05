"""Compare two normalizer loading paths on FRESH envs: cloudpickle vs VecNormalize.load."""
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import runtime_bootstrap
from env import build_env

config = json.load(open(CKPT + '/config.json'))
profile = config['throw_profile']
policy = PPO.load(CKPT + '/policy.zip', device='cpu')

def run(label, make_env, use_vecnorm):
    env = make_env()
    if use_vecnorm:
        vec = VecNormalize.load(CKPT + '/normalize.pkl', DummyVecEnv([lambda: env]))
        vec.training = False
        vec.norm_reward = False
        vec.reset()
        step_env = vec
    else:
        with open(CKPT + '/normalize.pkl', 'rb') as fh:
            norm = cloudpickle.load(fh)
        obs, _ = env.reset(seed=1)
        step_env = env
        normed = norm
    results = []
    for seed in (1, 2, 3):
        if use_vecnorm:
            obs = vec.reset()
        else:
            obs, _ = env.reset(seed=seed)
        done, steps = False, 0
        while not done:
            a = step_env.action_space.sample() * 0  # placeholder
            if use_vecnorm:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action, _ = policy.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
                action = action[0]
            if steps == 0:
                a0 = np.asarray(action).ravel()
                print(f'{label} seed{seed} action0[:3]={np.round(a0[:3], 3)} rel={a0[-1]:+.2f}')
            if use_vecnorm:
                obs, _, dones, infos = step_env.step(action)
                done = dones[0]
                inf = infos[0]
            else:
                obs, r, term, trunc, inf = step_env.step(action)
                done = term or trunc
            steps += 1
        results.append((inf.get('outcome'), inf.get('impact_error_m')))
    print(f'{label}:', results)

run('cloudpickle ', lambda: build_env(variant='arc_v7', throw_profile=profile, randomize=False, record=False), False)
run('VecNormalize', lambda: build_env(variant='arc_v7', throw_profile=profile, randomize=False, record=False), True)
