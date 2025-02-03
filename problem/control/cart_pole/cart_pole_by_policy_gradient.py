"""Practice of policy gradient algorithm on CartPole-v0 environment."""

import gymnasium as gym
import logging
import numpy as np

from agent.policy_gradient_net import SimplePolicyGradientNet
from pipeline.play import ReplayRecords, play


class CartPolePolicyGradientSolver:
    """A solver for cart pole problem with policy gradient."""

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = self.env.action_space.n
        self.policy_net = SimplePolicyGradientNet(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            mlp_hidden_dims=[16, 24, 32],
            learning_rate=1e-4,
            gamma=0.9,
            non_terminated_init_reward=100,
        )
        self.play_records = ReplayRecords()
        self.play_records.max_capacity = 1

    def play_and_learn(self, n_episodes=1000):
        for episode in range(n_episodes):
            play_record = play(self.env, self.policy_net.action_fn)
            self.play_records.receive_play_record(play_record)
            self.policy_net.update(self.play_records)
            if episode % 100 == 0:
                logging.info(f"Episode {episode}, rewards {sum(play_record.rewards)}")

    def play(self, n_episodes=10):
        env = gym.make("CartPole-v1", render_mode="human")
        for _ in range(n_episodes):
            play_record = play(env, self.policy_net.action_fn)
            logging.info(f"Play rewards {sum(play_record.rewards)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    solver = CartPolePolicyGradientSolver()
    solver.play_and_learn(n_episodes=10000)
    solver.play()
    logging.info("Done.")
