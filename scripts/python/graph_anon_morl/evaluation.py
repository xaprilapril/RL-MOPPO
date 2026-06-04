import numpy as np
import torch

from .utils import compute_k_anonymity_reward, compute_utility_reward


def compute_pareto_front(reward_vecs):
    """
    Return indices of Pareto-optimal points (maximization in both objectives).
    reward_vecs: list of (r_priv, r_util) tuples or array of shape (N, 2).

    Point i is Pareto-optimal iff no other point j dominates it
    (j dominates i: j >= i in ALL objectives AND j > i in AT LEAST ONE).
    """
    rewards = np.array(reward_vecs, dtype=np.float64)
    n = len(rewards)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        # Find all points that i dominates: i >= j in all, i > j in at least one
        i_dominates = (
            np.all(rewards[i] >= rewards, axis=1)
            & np.any(rewards[i] > rewards, axis=1)
        )
        i_dominates[i] = False          # never mark i itself
        is_pareto[i_dominates] = False  # those are dominated — remove them
    return np.where(is_pareto)[0]


def hypervolume_indicator(pareto_front, reference_point=(-1.0, -1.0)):
    """2-D hypervolume (maximization). Sweeps left-to-right above reference_point."""
    if not pareto_front:
        return 0.0
    front = sorted(pareto_front, key=lambda p: p[0])
    ref_x, ref_y = reference_point
    hv = 0.0
    prev_x = ref_x
    for px, py in front:
        if px > prev_x and py > ref_y:
            hv += (px - prev_x) * (py - ref_y)
            prev_x = px
    return hv


def sample_action_no_repeat(logits, already_flipped, temperature=1.0):
    """
    Sample action with temperature scaling, masking recently-repeated edges.
    This prevents the deterministic collapse where argmax always picks the same
    edge to flip back and forth (confirmed degenerate behavior on large action spaces).

    - temperature < 1.0 → more deterministic (sharper distribution)
    - temperature = 1.0 → standard softmax sampling
    - already_flipped: set of action indices flipped an odd number of times (currently ON)
      We down-weight these to avoid immediate undo flips.
    """
    logits_adj = logits.clone().float()
    if already_flipped:
        for a in already_flipped:
            logits_adj[0, a] -= 5.0   # heavy penalty for undoing a flip just made
    scaled = logits_adj / temperature
    probs = torch.softmax(scaled, dim=-1)
    action = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return action


def evaluate_policy(env, model, n_weights: int = 11, n_episodes: int = 5,
                    device: str = "cpu", temperature: float = 0.3):
    """
    Sweep α ∈ linspace(0,1,n_weights) and collect mean reward vectors.
    Returns list of (alpha, mean_r_priv, mean_r_util).

    Uses stochastic sampling (deterministic=False) averaged over n_episodes.
    Pure argmax (deterministic=True) collapses to a degenerate oscillating
    action on large action spaces before full policy convergence.
    Stochastic evaluation is the honest measure of the learned distribution.
    """
    results = []
    alphas = np.linspace(0.0, 1.0, n_weights)
    model.eval()
    with torch.no_grad():
        for alpha in alphas:
            ep_rewards = []
            for _ in range(n_episodes):
                obs, _ = env.reset(weight=[alpha, 1.0 - alpha])
                done = False
                while not done:
                    obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    mask_t = torch.BoolTensor(env.get_action_mask()).unsqueeze(0).to(device)
                    action, _, _, _ = model.get_action(obs_t, deterministic=False, mask=mask_t)
                    obs, _, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated
                final_rp = compute_k_anonymity_reward(env.G, env.k)
                final_ru = compute_utility_reward(env.G, env.G0)
                ep_rewards.append([final_rp, final_ru])
            mean_r = np.mean(ep_rewards, axis=0)
            results.append((float(alpha), float(mean_r[0]), float(mean_r[1])))
    return results
