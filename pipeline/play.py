"""Interact with the env and return records."""

import gymnasium as gym
import logging
import random
from typing import Any, List


class PlayRecord:
    """A record for one play."""

    def __init__(self):
        self.rewards: List[float] = []
        self.states: List[Any] = (
            []
        )  # Reuse for next state, len(states) = len(rewards) + 1
        self.is_terminated: bool = (
            False  # True for meet terminated state, False for truncated.
        )
        self.actions: List[Any] = []  # len(actions) + 1 == len(states)

    def append(self, reward, state, action, terminated):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.is_terminated = terminated


class RecordSamples:
    """Samples for batch learning."""

    def __init__(self):
        self.states: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.mc_rewards: List[float] = []
        self.next_states: List[Any] = []
        self.terminated: List[bool] = []  # Is terminated at next state.

    def __len__(self) -> int:
        return len(self.rewards)


class ReplayRecords:
    """A storage for records from plays."""

    def __init__(self):
        self.records: List[PlayRecord] = []
        self.max_capacity: int = 500

    def clear(self):
        self.records.clear()

    def receive_play_record(self, play_record: PlayRecord):
        self.records.append(play_record)
        while len(self.records) > self.max_capacity:
            self.records.pop(0)

    def sample(
        self,
        batch_size: int,
        gamma: float = 0.9,
        non_terminated_init_reward: float = 500,
    ) -> RecordSamples:
        assert len(self.records) > 0, "No records."
        idx = list(range(len(self.records)))
        random.shuffle(idx)
        samples = RecordSamples()
        while len(samples) < batch_size:
            for i in idx:
                record = self.records[i]
                samples.states += record.states[:-1]
                samples.actions += record.actions
                samples.rewards += record.rewards
                # Do monte carlo rewards calculation.
                mc_rewards = [0] * len(record.rewards)
                acc_r = 0 if record.is_terminated else non_terminated_init_reward
                for j in range(len(record.rewards) - 1, -1, -1):
                    acc_r *= gamma
                    acc_r += record.rewards[j]
                    mc_rewards[j] = acc_r
                samples.mc_rewards += mc_rewards
                samples.next_states += record.states[1:]
                term = [False] * len(record.rewards)
                term[-1] = record.is_terminated
                samples.terminated += term
        return samples


def play(env: gym.Env, action_fn: callable) -> PlayRecord:
    record = PlayRecord()
    record.states.append(env.reset()[0])
    while True:
        action = action_fn(record.states[-1])
        new_state, reward, terminated, truncated, _ = env.step(action)
        record.append(reward, new_state, action, terminated)
        if terminated or truncated:
            break
    return record


def _test_play():
    env = gym.make("Ant", render_mode="human")
    action_fn = lambda s: env.action_space.sample()
    records = ReplayRecords()
    for _ in range(3):
        records.receive_play_record(play(env, action_fn))
    samples = records.sample(batch_size=20)
    print(len(samples))


if __name__ == "__main__":
    _test_play()
