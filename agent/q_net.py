"""A QLearning network agent."""

import torch
from typing import List
import random

from model.module.mlp import MLP


class SimpleQNet:
    def __init__(self, state_dim: int, action_dim: int, mlp_hidden_dims: List[int]):
        self.action_dim = action_dim
        self.mlp = MLP([state_dim] + mlp_hidden_dims + [action_dim])

    def action_fn(self, state: torch.Tensor, epsilon=0.1) -> torch.Tensor:
        if random.random() < epsilon:
            return torch.randint(0, self.action_dim)
        return self.mlp(state)
