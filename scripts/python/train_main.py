"""MOPPO training for graph anonymization. Supports --resume for chunked runs."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from graph_anon_morl.datasets import load_facebook_ego
from graph_anon_morl.env import GraphAnonEnv
from graph_anon_morl.models import MOPPOActorCritic


def make_graph(name, n=20, seed=42):
    if name == "karate":
        return nx.karate_club_graph()
    if name == "barbell":
        return nx.barbell_graph(n // 2, 1)
    if name == "random":
        return nx.erdos_renyi_graph(n, 0.3, seed=seed)
    if name == "facebook":
        return load_facebook_ego(n_nodes=n, seed=seed)
    raise ValueError(f"Unknown graph: {name}")


def collect_rollout(env, model, n_steps, device, alpha_range=None, fixed_alpha=None):
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, mask_buf = [], [], [], [], [], [], []

    def reset_env():
        if fixed_alpha is not None:
            return env.reset(weight=[fixed_alpha, 1.0 - fixed_alpha])[0]
        opts = {"alpha_range": alpha_range} if alpha_range is not None else None
        return env.reset(options=opts)[0]

    obs = reset_env()
    for _ in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        mask_np = env.get_action_mask()
        mask_t  = torch.BoolTensor(mask_np).unsqueeze(0).to(device)
        with torch.no_grad():
            action, logp, _, value = model.get_action(obs_t, mask=mask_t)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        obs_buf.append(obs)
        act_buf.append(action.item())
        logp_buf.append(logp.item())
        val_buf.append(value.item())
        rew_buf.append(reward)
        done_buf.append(float(done))
        mask_buf.append(mask_np)
        obs = next_obs if not done else reset_env()
    return (
        torch.FloatTensor(np.array(obs_buf)).to(device),
        torch.LongTensor(act_buf).to(device),
        torch.FloatTensor(logp_buf).to(device),
        torch.FloatTensor(val_buf).to(device),
        torch.FloatTensor(rew_buf).to(device),
        torch.FloatTensor(done_buf).to(device),
        torch.BoolTensor(np.array(mask_buf)).to(device),
    )


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    n = len(rewards)
    advantages = torch.zeros(n)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = values[t + 1].item() if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta.item() + gamma * lam * (1 - dones[t].item()) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values.cpu()


def ppo_update(model, optimizer, obs, actions, old_logps, returns, advantages, masks,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.05, n_epochs=4, batch_size=64):
    n = len(obs)
    device = obs.device
    returns = returns.to(device)
    advantages = advantages.to(device)
    total_loss = 0.0
    for _ in range(n_epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]
            logits, values = model(obs[mb])
            # Must apply the same mask used at collection time so the ratio is correct
            logits = logits.masked_fill(~masks[mb], -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            logps = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()
            ratio = (logps - old_logps[mb]).exp()
            adv = advantages[mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            policy_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv,
            ).mean()
            value_loss = (values - returns[mb]).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
    return total_loss / (n_epochs * max(1, n // batch_size))


def try_load_checkpoint(path, device):
    """Try to load a checkpoint; return None if file is missing or corrupt."""
    p = Path(path)
    if not p.exists() or p.stat().st_size < 1000:
        return None
    try:
        return torch.load(p, map_location=device)
    except Exception:
        return None


def save_checkpoint(ckpt_path, metrics_path, model, optimizer,
                    obs_dim, env, iteration, metrics, args):
    """Save with .bak fallback: keep previous good save before overwriting."""
    bak = Path(str(ckpt_path) + ".bak")
    cur = Path(ckpt_path)
    # Promote current to backup before overwriting
    if cur.exists() and cur.stat().st_size > 1000:
        try:
            shutil.copy2(str(cur), str(bak))
        except Exception:
            pass
    # Direct save (may corrupt on kill, but .bak has previous good state)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "obs_dim": obs_dim,
        "n_nodes": env.n,
        "n_actions": env.n_actions,
        "completed_iters": iteration + 1,
        "args": vars(args),
    }, ckpt_path)
    # Save metrics directly (small file, fast write, low corruption risk)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    G0 = make_graph(args.graph, args.n_nodes, args.seed)
    print(f"Graph '{args.graph}': {G0.number_of_nodes()} nodes, {G0.number_of_edges()} edges")

    env = GraphAnonEnv(G0, k=args.k, T=args.episode_len)
    obs_dim = env.observation_space.shape[0]
    print(f"obs_dim={obs_dim}, n_actions={env.n_actions}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "moppo_checkpoint.pt"
    metrics_path = out_dir / "train_metrics.json"

    model = MOPPOActorCritic(obs_dim, env.n_actions, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_iter = 0
    metrics = []

    # Resume: try main checkpoint, then .bak
    if args.resume:
        ckpt = (try_load_checkpoint(ckpt_path, device) or
                try_load_checkpoint(str(ckpt_path) + ".bak", device))
        if ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_iter = ckpt.get("completed_iters", 0)
            try:
                content = Path(metrics_path).read_text().strip()
                metrics = json.loads(content) if content else []
            except Exception:
                metrics = []
            print(f"Resumed from iter {start_iter}/{args.n_iter}")

    if start_iter >= args.n_iter:
        print(f"Already complete ({start_iter}/{args.n_iter}). Nothing to do.")
        return model, env

    for iteration in range(start_iter, args.n_iter):
        # Fixed-alpha baselines: no curriculum, same α every episode
        fixed_alpha = args.fixed_alpha
        alpha_range = None
        if fixed_alpha is None and args.curriculum:
            if iteration < args.curriculum_phase1:
                alpha_range = (0.95, 1.0)
            elif iteration < args.curriculum_phase2:
                alpha_range = (0.0, 0.05)

        obs, acts, logps, vals, rews, dones, masks = collect_rollout(
            env, model, args.n_steps, device,
            alpha_range=alpha_range, fixed_alpha=fixed_alpha)
        advantages, returns = compute_gae(rews, vals, dones, args.gamma, args.lam)
        loss = ppo_update(model, optimizer, obs, acts, logps, returns, advantages, masks,
                          clip_eps=args.clip_eps)
        mean_rew = rews.mean().item()
        metrics.append({"iter": iteration, "loss": round(loss, 6),
                        "mean_reward": round(mean_rew, 6)})

        if (iteration + 1) % max(1, args.n_iter // 20) == 0:
            print(f"  [{iteration+1:4d}/{args.n_iter}] loss={loss:.4f}  mean_r={mean_rew:.4f}")

        if (iteration + 1) % 5 == 0 or (iteration + 1) == args.n_iter:
            save_checkpoint(ckpt_path, metrics_path, model, optimizer,
                            obs_dim, env, iteration, metrics, args)

    print(f"Done. Checkpoint -> {ckpt_path}  ({args.n_iter} iters)")
    return model, env


def parse_args():
    p = argparse.ArgumentParser(description="MOPPO for Graph Anonymization")
    p.add_argument("--graph", default="karate",
                   choices=["karate", "barbell", "random", "facebook"])
    p.add_argument("--n_nodes", type=int, default=20)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--episode_len", type=int, default=50)
    p.add_argument("--n_iter", type=int, default=2000)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="outputs/moppo")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint if it exists")
    p.add_argument("--curriculum", action="store_true",
                   help="Enable curriculum learning: privacy→utility→mixed phases")
    p.add_argument("--curriculum_phase1", type=int, default=100,
                   help="Iter to end Phase 1 (privacy-only, α~U(0.95,1))")
    p.add_argument("--curriculum_phase2", type=int, default=200,
                   help="Iter to end Phase 2 (utility-only, α~U(0,0.05))")
    p.add_argument("--fixed_alpha", type=float, default=None,
                   help="Fix alpha to a constant (0.0=utility-only, 1.0=privacy-only). Disables curriculum.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
