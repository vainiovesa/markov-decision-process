from random import random, choice


class Model:
    def __init__(self, gamma: float, epsilon: float, actions: list, learning_rate:float):
        """gamma: discount factor, epsilon: randomization factor"""
        assert 0 <= gamma <= 1
        assert 0 < learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.memory = {}
        self.lr = learning_rate

    def action_values(self, state):
        if state in self.memory:
            return self.memory[state]
        return [(0, 0) for _ in range(len(self.actions))]

    def action_value(self, state):
        action_values = self.action_values(state)
        values = []
        for ret, n in action_values:
            values.append(ret / n if n > 0 else 0)
        return values

    def memory_append(self, state, action: int, ret: float):
        action_values = self.action_values(state)
        returns, n = action_values[action]

        if n > 0:
            ret = ret - returns / n
        ret *= self.lr

        returns, n = returns + ret, n + 1
        action_values[action] = returns, n
        self.memory[state] = action_values

    def add_trajectory(self, states: list, actions: list, rewards: list):
        returns = _calculate_return(rewards, self.gamma)
        for state, action, ret in zip(states, actions, returns):
            self.memory_append(state, action, ret)

    def action(self, state):
        action_value = self.action_value(state)
        if random() < self.epsilon:
            return choice(self.actions)
        return action_value.index(max(action_value))


def _calculate_return(rewards: list, gamma: float):
    returns = []

    ret_t_minus_one = 0
    for i in range(len(rewards)):
        r_t = rewards[- i - 1]
        ret = r_t + gamma * ret_t_minus_one
        returns.append(ret)
        ret_t_minus_one = ret

    returns.reverse()
    return returns
