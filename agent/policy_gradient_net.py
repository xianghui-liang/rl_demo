"""A policy gradient network agent."""

import numpy as np
import torch
from typing import List

from model.module.mlp import MLP
from pipeline.play import ReplayRecords


class SimplePolicyGradientNet:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        mlp_hidden_dims: List[int],
        learning_rate: float,
        gamma: float,
        non_terminated_init_reward: float = 500,
    ):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.policy_net = MLP([state_dim] + mlp_hidden_dims + [action_dim]).to(
            self.device
        )
        self.policy_net_opt = torch.optim.AdamW(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.action_dim = action_dim
        self.gamma = gamma
        self.non_terminated_init_reward = non_terminated_init_reward

    def action_fn(self, state: torch.Tensor) -> int:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        state = state.to(self.device)
        with torch.no_grad():
            dist = torch.softmax(self.policy_net(state), dim=-1)
        return np.random.choice(self.action_dim, p=dist.cpu().numpy())

    def update(self, play_records: ReplayRecords):
        self.policy_net.train()
        self.policy_net_opt.zero_grad()
        loss = 0
        for record in play_records.records:
            rewards = record.rewards
            states = torch.tensor(record.states[:-1]).to(self.device)
            actions = record.actions
            mc_rewards = np.zeros_like(rewards, dtype=np.float32)
            mc_reward = (
                self.non_terminated_init_reward if not record.is_terminated else 0
            )
            for i in reversed(range(len(rewards))):
                mc_reward = rewards[i] + self.gamma * mc_reward
                mc_rewards[i] = mc_reward
            dist = torch.softmax(
                self.policy_net(torch.tensor(states)), dim=-1
            )  # [T, action_dim]
            for i, (action, mc_reward) in enumerate(zip(actions, mc_rewards)):
                loss -= torch.log(dist[i][action]) * mc_reward * (self.gamma**i)
        loss /= len(play_records.records)
        loss.backward()
        self.policy_net_opt.step()
