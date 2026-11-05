"""Render the best throws of a checkpoint to MP4 (wide view + bullseye close-up).

Runs the deterministic policy over N seeds, keeps the K throws with the smallest
miss distance among valid board hits, encodes each at 50 fps with ffmpeg
(threads 2). Offline rendering -- no realtime pacing, CPU-friendly.

Run:
  .venv/bin/python render_best_throws.py --checkpoint <ckpt-dir> --episodes 40 --keep 5
Output: demo/*.mp4 + demo/throws_summary.json
"""
import os, sys, json, time, argparse, subprocess
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
import mujoco
from stable_baselines3 import PPO

import runtime_bootstrap
from env import build_env

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--episodes', type=int, default=40)
ap.add_argument('--seed-start', type=int, default=92001)
ap.add_argument('--keep', type=int, default=5)
ap.add_argument('--randomize', action='store_true')
ap.add_argument('--profile-json', default=None, help='override the throw profile (e.g. the match-conditions centered aim)')
ap.add_argument('--out-dir', default='demo')
args = ap.parse_args()

ckpt = os.path.abspath(args.checkpoint)
config = json.load(open(os.path.join(ckpt, 'config.json')))
profile = config.get('throw_profile')
if profile is not None and 'profile' in profile:
    profile = profile['profile']
if args.profile_json is not None:
    profile = json.load(open(args.profile_json))
    print('profile override:', args.profile_json, flush=True)

policy = PPO.load(os.path.join(ckpt, 'policy.zip'), device='cpu')
with open(os.path.join(ckpt, 'normalize.pkl'), 'rb') as fh:
    norm = cloudpickle.load(fh)

env = build_env(variant='arc_v7', throw_profile=profile, randomize=args.randomize, record=True)
sim = env.sim

def render_frames(frames, camera, size=(800, 480)):
    renderer = mujoco.Renderer(sim.model, size[1], size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = camera
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0  # hide site group like the repo's render()
    out = []
    final_q, final_v = sim.data.qpos.copy(), sim.data.qvel.copy()
    try:
        for _, q, v in frames:
            sim.data.qpos[:] = q
            sim.data.qvel[:] = v
            mujoco.mj_forward(sim.model, sim.data)
            renderer.update_scene(sim.data, cam, scene_option=opt)
            out.append(renderer.render())
    finally:
        sim.data.qpos[:] = final_q
        sim.data.qvel[:] = final_v
        mujoco.mj_forward(sim.model, sim.data)
        renderer.close()
    return out

WIDE = ([1.5, 0., 1.45], 4.2, 90, -8)          # side view: robot -> board
CLOSE = ([2.95, 0., 1.73], 0.55, 90, 0)        # bullseye close-up

throws = []
t0 = time.perf_counter()
for seed in range(args.seed_start, args.seed_start + args.episodes):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = policy.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
        obs, _, term, trunc, info = env.step(action[0])
        done = term or trunc
    board = info.get('outcome') == 'board' and info.get('reward_terms', {}).get('board_impact', 0) > 0
    err = info.get('impact_error_m') if board else None
    throws.append({'seed': seed, 'outcome': info.get('outcome'), 'error_m': err,
                   'bullseye': bool(info.get('is_success')), 'frames': list(env.frames)})
    hits = sum(1 for t in throws if t['error_m'] is not None)
    print(f"seed {seed}: {info.get('outcome')} err={err}  [{hits} hits so far, {time.perf_counter()-t0:.0f}s]",
          flush=True)

env.close()
hits = sorted([t for t in throws if t['error_m'] is not None], key=lambda t: t['error_m'])[:args.keep]
os.makedirs(args.out_dir, exist_ok=True)
summary = []
for t in hits:
    frames = t.pop('frames')
    base = os.path.join(args.out_dir, f"throw_seed{t['seed']}_err{t['error_m']*100:.1f}cm")
    for suffix, camera in (('_wide', WIDE), ('_bullseye', CLOSE)):
        path = base + suffix + '.mp4'
        rgb = render_frames(frames, camera)
        pipe = subprocess.Popen(
            ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', '800x480', '-r', '50', '-i', '-', '-an', '-c:v', 'libx264',
             '-threads', '2', '-preset', 'veryfast', '-crf', '23',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
            stdin=subprocess.PIPE)
        for img in rgb:
            pipe.stdin.write(img.tobytes())
        pipe.stdin.close()
        if pipe.wait():
            raise RuntimeError('ffmpeg failed for ' + path)
        t[suffix + '_mp4'] = path
        print('wrote', path, flush=True)
    summary.append(t)

json.dump({'checkpoint': ckpt, 'episodes': args.episodes, 'seed_start': args.seed_start,
           'kept': summary,
           'all_hits': [{'seed': t['seed'], 'error_m': t['error_m']} for t in sorted(
               [x for x in throws if x['error_m'] is not None], key=lambda x: x['error_m'])]},
          open(os.path.join(args.out_dir, 'throws_summary.json'), 'w'), indent=2)
print('done ->', args.out_dir)
