import matplotlib.pyplot as plt
from sarsa_model import Model
from q_learning_model import QModel
from train import GridTrain, GridTrainHidden, train_sarsa as train


def create_agent(modeltype: str):
    actions = [0, 1, 2, 3]
    discount_factor = 0.99
    rand_factor = 0.6
    
    if modeltype == "sarsa":
        learning_rate = 0.1
        return Model(discount_factor, rand_factor, actions, learning_rate)
    
    learning_rate = 2
    return QModel(discount_factor, rand_factor, actions, learning_rate)


def example1(modeltype: str):
    agent = create_agent(modeltype)

    moves = 300
    hidden = GridTrainHidden(agent, moves=moves)
    info = train(hidden, 10000)
    plt.plot(info)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.show()

    visible = GridTrain(agent, moves=moves)
    visible.play()


def example2(modeltype: str):
    agent = create_agent(modeltype)

    moves = 300
    obstacles = []
    for i in range(1, 9):
        obstacles.append((i, 4))
    for i in range(1, 9):
        obstacles.append((8, i))
    for i in range(6, 10):
        obstacles.append((2, i))
    hidden = GridTrainHidden(agent, obstacles=obstacles, moves=moves)
    info = train(hidden, 15000)
    plt.plot(info)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.show()

    visible = GridTrain(agent, obstacles=obstacles, moves=moves)
    visible.play()


def example3(modeltype: str):
    agent = create_agent(modeltype)

    moves = 300
    obstacles = []
    for i in range(1, 9):
        obstacles.append((i, 1))
    for i in range(2, 9):
        obstacles.append((8, i))
    for i in range(4, 9):
        obstacles.append((1, i))
    for i in range(3, 10):
        obstacles.append((3, i))
    for i in range(4, 7):
        obstacles.append((i, 3))
    for i in range(5, 8):
        obstacles.append((i, 5))
    for i in range(5, 7):
        obstacles.append((i, 7))
        obstacles.append((i, 8))
    obstacles.append((1, 2))
    obstacles.append((9, 8))
    hidden = GridTrainHidden(agent, obstacles=obstacles, moves=moves)
    info = train(hidden, 15000)
    plt.plot(info)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.show()

    visible = GridTrain(agent, obstacles=obstacles, moves=moves)
    visible.play()


def main():
    example1("sarsa")
    example2("sarsa")
    example3("sarsa")


if __name__ == "__main__":
    main()
