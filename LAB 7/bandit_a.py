import numpy as np

epsilon = 0.1
num_actions = 2
num_iterations = 1000
Q = np.zeros(num_actions)
N = np.zeros(num_actions)


def binaryBanditA():
    return np.random.choice([1, 0], p=[0.5, 0.5])


def binaryBanditB():
    return np.random.choice([1, 0], p=[0.4, 0.6])


for i in range(num_iterations):
    if np.random.rand() < epsilon:
        action = np.random.choice(num_actions)
    else:
        action = np.argmax(Q)

    if action == 0:
        reward = binaryBanditA()
    else:
        reward = binaryBanditB()

    N[action] += 1
    Q[action] = Q[action] + (1 / N[action]) * (reward - Q[action])

best_action = np.argmax(Q)
print(f"The best action is: {'Bandit A' if best_action == 0 else 'Bandit B'}")
