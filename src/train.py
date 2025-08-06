from grid import GridGame
from sarsa_model import Model


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
            self.clock.tick(10)
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


def train_sarsa(grid: GridTrain, iterations: int):
    initial_rand_factor = grid.model.epsilon
    training_info = []
    for _ in range(iterations):
        states, actions, rewards = grid.play()
        training_info.append(sum(rewards))
        grid.model.add_trajectory(states, actions, rewards)
        grid.model.epsilon -= initial_rand_factor / iterations
    return training_info
