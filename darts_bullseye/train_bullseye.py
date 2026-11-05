"""Bullseye fine-tune trainer for the G1 five-finger dart thrower.

Slim Mac-friendly re-implementation of the repo's experiment2/train.py training
path with two deliberate differences:

1. NO per-episode Archive wrapper (their Archive renders an mp4 with ffmpeg for
   every completed throw -- the single biggest CPU/heat cost on a laptop).
   Periodic physical evaluations still run via their physics_eval.PhysicsEvaluation,
   using a pass-through identity wrapper instead of the video archive.
2. A bullseye-focused bonus added at episode end on top of the repo's measured
   Lorentzian target reward (profile target_reward_weight): a tight bonus for
   valid board hits close to the 6.35 mm inner bull, and a smaller bonus for
   clean throws that at least crossed the board plane.

Everything else mirrors their proven arc_v7 fine-tune recipe (resume from
best_verified_20261031, frozen observation statistics, LR 1e-4, entropy 0.001,
exploration std 0.2, rollout 128, domain randomization on).

Run (gentle chunk example):
  .venv/bin/python train_bullseye.py --steps 100000 --workers 2
Stop cleanly early:  touch <run-root>/STOP   (or kill -TERM the process)
"""
import os, sys, json, time, signal, argparse
from collections import deque, Counter
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

REPO = '/Users/saichaitu/Desktop/humaniod/G1-Throw-Darts/lab'
for p in (REPO, REPO + '/experiment2'):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
torch.set_num_threads(1)
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import FloatSchedule

import runtime_bootstrap  # their import guard, harmless on macOS


class BullseyeBonus(gym.Wrapper):
    """Terminal accuracy bonus on top of the repo's reward.

    board hit (valid, tip-first, clean):  board_bonus / (1 + (err/board_width)^2)
    clean throw crossing the board plane: cross_bonus / (1 + (err/cross_width)^2)
    """

    def __init__(self, env, board_bonus=24., board_width=.004, cross_bonus=6., cross_width=.01):
        super().__init__(env)
        self.board_bonus = float(board_bonus)
        self.board_width = float(board_width)
        self.cross_bonus = float(cross_bonus)
        self.cross_width = float(cross_width)

    def bonus(self, info):
        terms = info.get('reward_terms') or {}
        if (info.get('outcome') == 'board' and terms.get('board_impact', 0) > 0
                and info.get('impact_error_m') is not None):
            e = info['impact_error_m']
            return self.board_bonus / (1 + (e / self.board_width) ** 2)
        if (info.get('outcome') in ('environment_hit', 'timeout')
                and info.get('release') is not None and info.get('board_plane_crossing')):
            e = info['board_plane_crossing']['error_m']
            return self.cross_bonus / (1 + (e / self.cross_width) ** 2)
        return 0.

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if term or trunc:
            b = self.bonus(info)
            if b:
                reward += b
                info.setdefault('reward_terms', {})['bullseye_bonus'] = b
        return obs, reward, term, trunc, info


class IdentityArchive(gym.Wrapper):
    """Drop-in for train.Archive that records nothing (used by physics_eval)."""

    def __init__(self, env, folder):
        super().__init__(env)


def factory(run_root, rank, randomize, variant, profile, board_bonus, board_width, cross_bonus, cross_width):
    def create():
        from env import build_env
        raw = BullseyeBonus(
            build_env(variant=variant, throw_profile=profile, randomize=randomize, record=False),
            board_bonus, board_width, cross_bonus, cross_width)
        return Monitor(raw, str(Path(run_root) / f'monitor_{rank}.csv'))
    return create


class Progress(BaseCallback):
    """Lightweight progress.json + STOP-file abort."""

    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.rows = deque(maxlen=200)
        self.completed = 0

    def _on_training_start(self):
        self.write()

    def _on_rollout_start(self):
        self.write()

    def _on_step(self):
        for done, info in zip(self.locals.get('dones', ()), self.locals.get('infos', ())):
            if done:
                self.completed += 1
                self.rows.append({'outcome': info.get('outcome'),
                                  'bullseye': bool(info.get('is_success')),
                                  'error_m': info.get('impact_error_m')})
        return not (self.root / 'STOP').exists()

    def write(self):
        rows = list(self.rows)
        errs = [r['error_m'] for r in rows if r['error_m'] is not None]
        data = {
            'timesteps': self.num_timesteps,
            'completed_episodes': self.completed,
            'outcomes_last200': dict(Counter(str(r['outcome']) for r in rows)),
            'bullseyes_last200': sum(r['bullseye'] for r in rows),
            'board_hits_last200': sum(r['outcome'] == 'board' for r in rows),
            'median_error_last200_m': round(float(np.median(errs)), 4) if errs else None,
        }
        tmp = self.root / 'progress.tmp'
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.root / 'progress.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=REPO + '/experiment2/models/best_verified_20261031')
    ap.add_argument('--steps', type=int, default=100_000, help='additional timesteps this chunk')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--rollout', type=int, default=128)
    ap.add_argument('--seed', type=int, default=20261105)
    ap.add_argument('--randomize', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--entropy', type=float, default=0.001)
    ap.add_argument('--exploration-std', type=float, default=0.2)
    ap.add_argument('--clip-range', type=float, default=0.2)
    ap.add_argument('--checkpoint-every', type=int, default=2048)
    ap.add_argument('--eval-every', type=int, default=8192)
    ap.add_argument('--eval-episodes', type=int, default=12)
    ap.add_argument('--eval-seed', type=int, default=91001)
    ap.add_argument('--board-bonus', type=float, default=24.)
    ap.add_argument('--board-width', type=float, default=.004)
    ap.add_argument('--cross-bonus', type=float, default=6.)
    ap.add_argument('--cross-width', type=float, default=.01)
    ap.add_argument('--run-root', type=Path, default=Path('runs'))
    ap.add_argument('--profile-json', type=Path, default=None,
                    help='Optional throw-profile JSON (dict of profile params) overriding the checkpoint profile')
    args = ap.parse_args()

    def stop(signum, frame):
        raise SystemExit(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)

    source = Path(os.path.abspath(args.checkpoint))
    config = json.loads((source / 'config.json').read_text())
    variant = 'arc_v7'  # profiles are versioned against arc_v7 (build_env enforces it)
    profile = config.get('throw_profile')
    if profile is not None and 'profile' in profile:
        profile = profile['profile']
    if args.profile_json is not None:
        profile = json.loads(Path(args.profile_json).read_text())
        print('overriding throw profile from', args.profile_json, flush=True)
    assert profile is not None, 'checkpoint has no throw_profile; this trainer expects the calibrated arc_v7 stack'

    root = (args.run_root.resolve() / ('finetune_' + time.strftime('%Y%m%d_%H%M%S')))
    root.mkdir(parents=True)
    run_config = {'variant': variant, 'throw_profile': profile, 'source_checkpoint': str(source),
                  'additional_steps': args.steps, 'workers': args.workers, 'rollout': args.rollout,
                  'seed': args.seed, 'randomize': args.randomize, 'learning_rate': args.learning_rate,
                  'entropy': args.entropy, 'exploration_std': args.exploration_std,
                  'clip_range': args.clip_range, 'checkpoint_every': args.checkpoint_every,
                  'eval_every': args.eval_every, 'eval_episodes': args.eval_episodes,
                  'eval_seed': args.eval_seed, 'bullseye_bonus': {'board': args.board_bonus,
                  'board_width': args.board_width, 'cross': args.cross_bonus, 'cross_width': args.cross_width}}
    (root / 'config.json').write_text(json.dumps(run_config, indent=2))
    print('RUN ' + str(root), flush=True)

    venv = SubprocVecEnv([factory(root, i, args.randomize, variant, profile,
                                  args.board_bonus, args.board_width, args.cross_bonus, args.cross_width)
                          for i in range(args.workers)])
    vector = VecNormalize.load(str(source / 'normalize.pkl'), venv)
    vector.training = False          # freeze observation statistics (their fine-tune recipe)
    vector.norm_reward = False
    vector.seed(args.seed)

    policy = PPO.load(str(source / 'policy.zip'), env=vector, device='cpu', seed=args.seed,
                      n_steps=args.rollout, batch_size=min(256, args.rollout * args.workers))
    if args.learning_rate:
        policy.learning_rate = args.learning_rate
        policy.lr_schedule = FloatSchedule(args.learning_rate)
    if args.clip_range:
        policy.clip_range = FloatSchedule(args.clip_range)
    if args.entropy is not None:
        policy.ent_coef = args.entropy
    if args.exploration_std:
        with torch.no_grad():
            policy.policy.log_std.fill_(np.log(args.exploration_std))
        policy.policy.optimizer.state.pop(policy.policy.log_std, None)

    policy.set_logger(configure(str(root / 'logs'), ['stdout', 'csv']))

    from physics_eval import PhysicsEvaluation  # needs the sys.path setup done above
    (root / 'checkpoints').mkdir(exist_ok=True)
    callbacks = [
        CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.workers),
                           save_path=str(root / 'checkpoints'), save_vecnormalize=True),
        PhysicsEvaluation(root, variant, args.randomize, args.eval_every, args.eval_episodes,
                          args.eval_seed, IdentityArchive),
        Progress(root),
    ]

    failure = None
    started = time.perf_counter()
    try:
        policy.learn(args.steps, callback=callbacks, reset_num_timesteps=False)
        status = 'stopped' if (root / 'STOP').exists() else 'completed'
    except BaseException as error:  # noqa: BLE001 - mirror repo: save on any failure
        failure = type(error).__name__ + ': ' + error
        status = 'failed'
        raise
    finally:
        policy.save(str(root / 'policy'))
        vector.save(str(root / 'normalize.pkl'))
        (root / 'training_summary.json').write_text(json.dumps({
            'timesteps': policy.num_timesteps, 'additional_steps': args.steps,
            'elapsed_s': time.perf_counter() - started, 'status': status, 'error': failure}, indent=2))
        print(f'DONE {root} status={status} timesteps={policy.num_timesteps}', flush=True)
    venv.close()


if __name__ == '__main__':
    main()
