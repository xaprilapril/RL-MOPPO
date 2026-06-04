import torch
import torch.nn as nn


class MOPPOActorCritic(nn.Module):
    """
    Weight-conditioned actor-critic for MOPPO.

    Input:  degree_seq (n) + anon_deficit (n) + weight (2,)  →  obs_dim = 2n + 2
    Output: action logits (n_actions,) and state value (1,)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, n_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False,
                   mask: torch.Tensor = None):
        """
        mask: bool tensor (batch, n_actions) or (n_actions,).
              True = valid action. Invalid actions get logit = -1e9.
        """
        logits, value = self.forward(obs)
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
