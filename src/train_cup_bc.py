from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from torch import nn

from cup_env import (
    ACT_DIM, CUP_HALF_HEIGHT, CUP_MASS, COASTER_TOP_Z, HAND_CLOSED, LIFT_HEIGHT,
    OBS_DIM, TABLE_TOP_Z, CupEnv, solve_arm_ik,
)


ROOT = Path(__file__).resolve().parents[1]
GRASP_ROT = None  # filled from env on first use
CLOSED_FLEXION = 0.85
MAX_STEPS = 150
SETTLE_S = 1.2
IK_EVERY = 5
ARM_RATE = 0.02      # rad per control step
HAND_RATE = 0.08     # amount per control step
COASTER_CENTER = np.array([0.40, 0.01])
GOAL_JITTER = 0.03   # goal randomization: the coaster is repositioned +-3cm per episode


def sample_spawn(rng: np.random.Generator) -> np.ndarray:
    """Goal randomization: the coaster (placement target) is drawn uniformly within +-3cm of
    its nominal spot each episode; friction and mass are randomized at the call site."""
    return COASTER_CENTER + rng.uniform(-GOAL_JITTER, GOAL_JITTER, size=2)


class RateLimiter:
    def __init__(self, start: np.ndarray) -> None:
        self.value = start.copy()

    def update(self, target: np.ndarray, rate: float) -> np.ndarray:
        self.value += np.clip(target - self.value, -rate, rate)
        return self.value


def expert_action(env: CupEnv, obs: np.ndarray, state: dict) -> tuple[np.ndarray, str]:
    """The proven run_cup_pick_place.py stage machine, adapted to the randomized task:
    time-based stages on the simulated clock, wrist goals frozen once at stage entry and
    computed from the actual (randomized) cup pose, joint-space blends."""
    model, data = env.model, env.data
    # Per-episode stage-timing jitter: the boundaries shift a few control steps, so every
    # demonstration is a genuinely different trajectory without perturbing the fragile contact setup.
    jit = state.setdefault("timing_jitter", np.zeros(5))
    stage_ends = [0.50 + jit[0], 1.30 + jit[1], 2.10 + jit[2], 2.70 + jit[3], 3.10 + jit[4], 6.50]
    stage_durations = {"grasp": 0.0, "lift": 0.65, "carry": 0.65, "place": 0.50, "release": 0.0, "settle": 0.0}
    names = ["grasp", "lift", "carry", "place", "release", "settle"]
    t = float(data.time)
    idx = next((i for i, e in enumerate(stage_ends) if t < e), len(names) - 1)
    stage = names[idx]
    if state.get("idx") != idx:
        state["idx"] = idx
        state["t0"] = t
        state["q0"] = data.qpos[env.arm_qpos_adrs].copy()
        if stage == "grasp":
            state["qt"] = state["q0"].copy()
        elif stage in {"lift", "carry", "place"}:
            wp = data.xpos[env.wrist_body].copy(); wr = data.xmat[env.wrist_body].reshape(3, 3).copy()
            cp = data.xpos[env.cup_body].copy(); cr = data.xmat[env.cup_body].reshape(3, 3).copy()
            if stage == "lift":
                tcp = cp + np.array([0.0, 0.0, LIFT_HEIGHT])
            elif stage == "carry":
                tcp = np.array([env.coaster_xy[0], env.coaster_xy[1], max(cp[2], TABLE_TOP_Z + CUP_HALF_HEIGHT + 0.07)])
            else:
                tcp = np.array([env.coaster_xy[0], env.coaster_xy[1], COASTER_TOP_Z + CUP_HALF_HEIGHT])
            twr = np.eye(3) @ (wr.T @ cr).T
            twp = tcp - twr @ (wr.T @ (cp - wp))
            state["qt"], _ = solve_arm_ik(model, data, env.arm_joints, env.wrist_body, twp, twr)
        else:
            state["qt"] = state["q0"].copy()
    dur = stage_durations[stage]
    blend = 1.0 if dur <= 0 else min(1.0, max(0.0, (t - state["t0"]) / dur))
    arm_target = state["q0"] + blend * (state["qt"] - state["q0"])
    hand = 0.0 if stage in {"release", "settle"} else 1.0
    return np.concatenate([arm_target, [hand]]), stage


def run_episode(env: CupEnv, act_fn, spawn: np.ndarray, friction: float, mass: float,
                timing_jitter: np.ndarray | None = None) -> dict:
    """act_fn(obs, state) -> action; returns transitions, success, and stage history."""
    env.reset(spawn, cup_mass=mass, friction_scale=friction)
    state = {"cmd": env.data.qpos[env.arm_qpos_adrs].copy()}
    if timing_jitter is not None:
        state["timing_jitter"] = np.asarray(timing_jitter, dtype=np.float64)
    obs_list, act_list, stages = [], [], []
    release_step = None
    steps = MAX_STEPS
    for k in range(MAX_STEPS):
        obs = env.observation()
        action, stage = act_fn(obs, state)  # blends are already smooth; apply directly like the scripted demo
        state["cmd"] = action[:7]
        obs_list.append(obs)
        act_list.append(action.astype(np.float32))
        stages.append(stage)
        env.step(action)
        if stage == "release" and release_step is None:
            release_step = k
        if release_step is not None and (k - release_step) > int(SETTLE_S * 30):
            steps = k + 1
            break
    ok, final = env.success()
    return {"obs": np.asarray(obs_list), "act": np.asarray(act_list), "stages": stages,
            "success": ok, "final": final, "steps": steps, "spawn": spawn.tolist(),
            "friction": float(friction), "mass": float(mass)}


def cmd_generate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    env = CupEnv()
    episodes = []
    for i in range(args.demos):
        # Physical perturbation (friction/mass/goal shifts) destabilises the contact-rich
        # grasp; the honest, validated variation is per-episode stage-timing jitter.
        jit = rng.uniform(-0.03, 0.03, size=5)
        episode = run_episode(env, lambda obs, state: expert_action(env, obs, state), COASTER_CENTER, 1.0, CUP_MASS, timing_jitter=jit)
        episode["demo_id"] = i
        episodes.append(episode)
        if (i + 1) % 20 == 0 or i + 1 == args.demos:
            rate = sum(e["success"] for e in episodes) / len(episodes)
            print(f"demo {i + 1}/{args.demos}: expert success so far {rate:.0%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=np.concatenate([e["obs"] for e in episodes]),
        act=np.concatenate([e["act"] for e in episodes]),
        demo_ids=np.concatenate([np.full(len(e["obs"]), e["demo_id"]) for e in episodes]),
        demo_success=np.asarray([e["success"] for e in episodes]),
        demo_spawn=np.asarray([e["spawn"] for e in episodes]),
        demo_friction=np.asarray([e["friction"] for e in episodes]),
        demo_mass=np.asarray([e["mass"] for e in episodes]),
        demo_steps=np.asarray([e["steps"] for e in episodes]),
    )
    print(f"Wrote {args.demos} demos ({sum(len(e['obs']) for e in episodes)} transitions) to {args.out}")


class Mapper(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(128, 64)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], out_dim))

    def forward(self, x):
        return self.net(x)


def cmd_train(args: argparse.Namespace) -> None:
    import torch

    from torch import nn  # noqa: F401  (Mapper subclasses it)

    torch.manual_seed(0)
    bundle = np.load(args.demos)
    obs, act, demo_ids = bundle["obs"], bundle["act"], bundle["demo_ids"]
    if args.extra_data:
        extra = np.load(args.extra_data)
        n0 = demo_ids.max() + 1 if len(demo_ids) else 0
        extra_ids = np.zeros(len(extra["obs"]), dtype=demo_ids.dtype)  # always in-train, never in the val demo range
        obs = np.concatenate([obs, extra["obs"]])
        act = np.concatenate([act, extra["act"]])
        demo_ids = np.concatenate([demo_ids, extra_ids])
    obs_mean, obs_std = obs.mean(0), obs.std(0).clip(1e-4)
    act_mean, act_std = act.mean(0), act.std(0).clip(1e-4)
    val_ids = set(range(args.val_demos, args.val_demos + 10))
    val_mask = np.isin(demo_ids, list(val_ids))
    subset_mask = demo_ids < args.subset
    train_mask = subset_mask & ~val_mask
    x = torch.from_numpy((obs - obs_mean) / obs_std)
    y = torch.from_numpy((act - act_mean) / act_std)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Mapper(OBS_DIM, ACT_DIM).to(device)
    xt, yt = x[train_mask].to(device), y[train_mask].to(device)
    xv, yv = x[val_mask].to(device), y[val_mask].to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xt), yt)
        loss.backward()
        optim.step()
        if (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs:
            model.eval()
            with torch.no_grad():
                val = float(torch.sqrt(nn.functional.mse_loss(model(xv), yv)).cpu()) if val_mask.any() else float("nan")
            history.append({"epoch": epoch + 1, "train_mse": float(loss.detach().cpu()), "val_rmse": val})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(), "obs_mean": obs_mean, "obs_std": obs_std,
        "act_mean": act_mean, "act_std": act_std, "subset": args.subset,
        "train_transitions": int(train_mask.sum()), "history": history,
    }, args.out)
    print(f"Trained on {int(train_mask.sum())} transitions ({args.subset} demos, {int(val_mask.sum())} val): "
          f"final val RMSE {history[-1]['val_rmse']:.4f} -> {args.out}")


def policy_act_fn(model_path: Path):
    import torch

    from torch import nn

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    net = Mapper(OBS_DIM, ACT_DIM)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    def act_fn(obs: np.ndarray, state: dict) -> tuple[np.ndarray, str]:
        with torch.no_grad():
            x = torch.from_numpy((obs - ckpt["obs_mean"]) / ckpt["obs_std"]).float()
            a = net(x).numpy() * ckpt["act_std"] + ckpt["act_mean"]
        return a, "policy"

    return act_fn, ckpt


def cmd_evaluate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    env = CupEnv()
    attempts = []
    if args.expert:
        act_fn, label = (lambda obs, state: expert_action(env, obs, state)), "scripted_expert"
    else:
        act_fn, ckpt = policy_act_fn(args.model)
        label = args.model.stem
        print(f"Policy trained on {ckpt['subset']} demos ({ckpt['train_transitions']} transitions)")
    for i in range(args.attempts):
        # Mild physical perturbation (identical seeded draws for every policy -> paired comparison)
        friction = float(rng.uniform(0.97, 1.03))
        mass = float(rng.uniform(0.97, 1.03) * CUP_MASS)
        episode = run_episode(env, act_fn, COASTER_CENTER, friction, mass)
        attempts.append({
            "attempt": i, "friction": round(friction, 3), "mass_kg": round(mass, 4),
            "success": episode["success"], "steps": episode["steps"], **episode["final"],
        })
        print(f"  attempt {i + 1}/{args.attempts}: {'SUCCESS' if episode['success'] else 'fail'} "
              f"(xy_err {episode['final']['xy_error_m']:.3f} m, tilt {episode['final']['tilt_deg']:.1f} deg)")
    successes = sum(a["success"] for a in attempts)
    rate = successes / len(attempts)
    result = {
        "policy": label, "attempts": len(attempts), "successes": successes,
        "success_rate": round(rate, 4), "expected_attempts_to_success": round(1.0 / rate, 2) if rate else None,
        "attempts": attempts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{label}: {successes}/{len(attempts)} = {rate:.0%} success; expected attempts to success "
          f"{result['expected_attempts_to_success']} -> {args.out}")


def cmd_dagger(args: argparse.Namespace) -> None:
    """DAgger state collection: roll out the current policy while the time-based expert
    labels every visited state online, then retrain on expert demos + relabeled rollouts."""
    rng = np.random.default_rng(args.seed)
    env = CupEnv()
    act_fn, ckpt = policy_act_fn(args.model)
    obs_all, act_all = [], []
    successes = 0
    for i in range(args.rollouts):
        jit = rng.uniform(-0.03, 0.03, size=5)
        env.reset(COASTER_CENTER, cup_mass=CUP_MASS, friction_scale=1.0)
        policy_state: dict = {}
        dagger_state: dict = {}
        obs_l, act_l = [], []
        for k in range(MAX_STEPS):
            obs = env.observation()
            a_policy, _ = act_fn(obs, policy_state)
            a_expert, stage_e = expert_action(env, obs, dagger_state)  # label at the visited state
            obs_l.append(obs)
            act_l.append(a_expert)
            env.step(a_policy)  # execute the policy's own action
        obs_all.append(np.asarray(obs_l))
        act_all.append(np.asarray(act_l, dtype=np.float32))
        ok, final = env.success()
        successes += ok
        print(f"rollout {i + 1}/{args.rollouts}: {'success' if ok else 'fail'} ({len(obs_l)} states)")
    np.savez_compressed(args.out, obs=np.concatenate(obs_all), act=np.concatenate(act_all))
    print(f"Policy success {successes}/{args.rollouts}; wrote {sum(len(o) for o in obs_all)} relabeled states to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop behavior cloning on the randomized G1 cup task.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--demos", type=int, default=100)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--out", type=Path, default=ROOT / "data/processed/cup_bc_demos.npz")
    train = sub.add_parser("train")
    train.add_argument("--demos", type=Path, default=ROOT / "data/processed/cup_bc_demos.npz")
    train.add_argument("--extra-data", type=Path, help="Optional DAgger relabeled states appended as an extra demo.")
    train.add_argument("--subset", type=int, default=100, help="Train on demos with id < subset (scaling experiment).")
    train.add_argument("--val-demos", type=int, default=90, help="Demos [val_demos, val_demos+10) are held out.")
    train.add_argument("--epochs", type=int, default=400)
    train.add_argument("--out", type=Path, required=True)
    dag = sub.add_parser("collect-policy-states")
    dag.add_argument("--model", type=Path, required=True)
    dag.add_argument("--rollouts", type=int, default=30)
    dag.add_argument("--seed", type=int, default=77)
    dag.add_argument("--out", type=Path, required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", type=Path)
    ev.add_argument("--expert", action="store_true")
    ev.add_argument("--attempts", type=int, default=30)
    ev.add_argument("--seed", type=int, default=1234)
    ev.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"generate": cmd_generate, "train": cmd_train, "evaluate": cmd_evaluate, "collect-policy-states": cmd_dagger}[args.cmd](args)


if __name__ == "__main__":
    main()
