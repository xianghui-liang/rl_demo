"""A Q-Learning network agent."""

import torch
from typing import List

from model.module.mlp import MLP
from pipeline.play import RecordSamples


class SimpleQNet:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        mlp_hidden_dims: List[int],
        learning_rate: float,
    ):
        self.q_net = MLP([state_dim] + mlp_hidden_dims + [action_dim])
        self.q_net_opt = torch.optim.AdamW(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

    def action_fn(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.q_net(state)

    def update(self, samples: RecordSamples):
        self.q_net.train()
        q_vals = self.q_net(torch.tensor(samples.states)).gather(
            dim=-1, index=torch.tensor(samples.actions, dtype=torch.int64)
        )
        loss = self.loss_fn(q_vals, torch.tensor(samples.mc_rewards))
        self.q_net_opt.zero_grad()
        loss.backward()
        self.q_net_opt.step()
