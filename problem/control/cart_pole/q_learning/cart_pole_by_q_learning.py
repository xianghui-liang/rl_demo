"""Practice cart pole with Q-Learning."""

import gymnasium as gym
import numpy as np

from agent.q_net import SimpleQNet
from pipeline.play import ReplayRecords, play


class CartPoleQLearningSolver:
    """A solver for cart pole problem with QLearning."""

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = self.env.action_space.n
        self.q_net = SimpleQNet(
            self.state_dim,
            self.action_dim,
            mlp_hidden_dims=[16, 32, 64],
            learning_rate=1e-5,
        )
        self.play_records = ReplayRecords()

    def play_and_learn(self):
        pass


def main():
    solver = CartPoleQLearningSolver()
    for _ in range(100):
        solver.play_and_learn()
