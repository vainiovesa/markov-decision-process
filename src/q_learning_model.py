from random import random, choice
from sarsa_model import Model


class QModel(Model):
    def add_trajectory(self, states: list, actions: list, rewards: list):
        returns = _calculate_return(rewards, states, self.memory, self.gamma)
        for state, action, ret in zip(states, actions, returns):
            self.memory_append(state, action, ret)


def _calculate_return(rewards: list, states: list, memory: dict, gamma: float):
    returns = []

    for i in range(len(states) - 1):
        state = states[i + 1]
        reward = rewards[i]

        q = 0
        if state in memory:
            qs = memory[state]
            qs = [ret / n for ret, n in qs if n > 0]
            q = max(qs)
        ret = reward + gamma * q
        returns.append(ret)
    returns.append(rewards[-1])

    return returns
