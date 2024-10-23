import numpy as np
import matplotlib.pyplot as plt


num_arms = 10
iterations = 10000
explore_prob = 0.1
learning_rate = 0.1


true_value_estimates = np.zeros(num_arms)
reward_estimates = np.zeros(num_arms)
times_action_selected = np.zeros(num_arms)
cumulative_reward_tracker = []


for iter in range(iterations):

    true_value_estimates += np.random.normal(0, 0.01, num_arms)


    if np.random.rand() < explore_prob:
        chosen_action = np.random.randint(num_arms)
    else:
        chosen_action = np.argmax(reward_estimates)


    reward = np.random.normal(true_value_estimates[chosen_action], 1)


    reward_estimates[chosen_action] += learning_rate * (reward - reward_estimates[chosen_action])

  
    if iter == 0:
        cumulative_reward_tracker.append(reward)
    else:
        cumulative_reward_tracker.append(cumulative_reward_tracker[-1] + reward)


plt.plot(cumulative_reward_tracker)
plt.xlabel('Iterations')
plt.ylabel('Cumulative Reward')
plt.title('Performance of Epsilon-Greedy with Learning Rate')
plt.show()
