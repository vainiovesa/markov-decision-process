from grid import GridGame
from random import random, choice


class Model:
    def __init__(self, gamma: float, epsilon: float, actions: list):
        """gamma: discount factor, epsilon: randomization factor"""
        assert 0 <= gamma <= 1
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.memory = {}

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
    returns = [rewards[-1]]
    rewards = list(reversed(rewards[:-1]))

    for i, reward in enumerate(rewards):
        last_ret = returns[-1]
        returns.append(last_ret + gamma ** i * reward)

    returns.reverse()
    return returns


class GridTrain(GridGame):
    def __init__(self,
                 model: Model,
                 width = 10,
                 height = 10,
                 obstacles = [],
                 playerxy = (0, 0),
                 goal = None,
                 moves = None):
        super().__init__(width, height, obstacles, playerxy, goal, moves)
        self.obstacles = obstacles
        self.initial_player_xy = playerxy
        self.model = model
        self.fps = 1

    def actions(self):
        state = self.playerxy
        return self.model.action(state)

    def restart(self):
        self.init_pygame()
        self.__init__(self.model,
                      self.width,
                      self.height,
                      self.obstacles,
                      self.initial_player_xy,
                      self.goal,
                      self.maxmoves)

    def play(self):
        self.restart()
        self.run = True
        states = []
        actions = []
        rewards = []
        while self.run and self.moves < self.maxmoves:
            self.draw()
            states.append(self.playerxy)
            action = self.actions()
            actions.append(action)
            self.move(action)
            if self.win():
                print("win")
                rewards.append(0)
                self.run = False
            else:
                rewards.append(-1)
            self.clock.tick(60)
        self.stop()
        return states, actions, rewards


class GridTrainHidden(GridTrain):
    def restart(self):
        self.__init__(self.model,
                      self.width,
                      self.height,
                      self.obstacles,
                      self.initial_player_xy,
                      self.goal,
                      self.maxmoves)

    def play(self):
        self.restart()
        self.run = True
        states = []
        actions = []
        rewards = []
        while self.run and self.moves < self.maxmoves:
            states.append(self.playerxy)
            action = self.actions()
            actions.append(action)
            self.move(action)
            if self.win():
                rewards.append(0)
                self.run = False
            else:
                rewards.append(-1)
        return states, actions, rewards
