# Markov decision process

### Overview
A reinforcement learning technique for solving a maze game. The agent begins by aimlessly running around while keeping track of average return on each run.
The agent receives a reward of $-1$ for each move, except when reaching the goal, which yields a reward of $0$. Depending on the randomization factor $\epsilon$ the
agent either explores i.e. makes a random move or exploits i.e. follows the path of maximum expected reward.

The basis for this project can be found in [this remarkable video](https://www.youtube.com/watch?v=VnpRp7ZglfA&t=1709s).

### Technicalities
The expected return of each run is calculated by
```math
G_t=r_t+\gamma\;r_{t+1}+\gamma^2\;r_{t+2}+\gamma^3\;r_{t+3}+\dots
```
where $G_t$ is the expected return for the taken action at step $t$, $\gamma$ is a discount factor such that $0\le\gamma\le1$,
and $r_t$ is the reward for the taken action.

These values are then stored in the agent's memory for building an optimal state-action policy.

### Instructions
Clone the repository and install dependencies
```console
$ git clone git@github.com:vainiovesa/markov-decision-process.git
$ cd markov-decision-process
$ poetry install --no-root
```
Show examples
```console
$ poetry run python3 main.py
```
