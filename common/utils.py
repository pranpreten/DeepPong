import numpy as np
import matplotlib.pyplot as plt


def argmax(xs):
    idxes = [i for i, x in enumerate(xs) if x == max(xs)]
    if len(idxes) == 1:
        return idxes[0]
    elif len(idxes) == 0:
        return np.random.choice(len(xs))

    selected = np.random.choice(idxes)
    return selected


def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = argmax(qs)  # OR np.argmax(qs)
    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs


def plot_total_reward(reward_history):
    import numpy as np
    import matplotlib.pyplot as plt

    def moving_average(x, window=100):
        return np.convolve(x, np.ones(window)/window, mode='valid')

    # 첫 번째 그래프: Raw Reward
    plt.figure(figsize=(10, 5))
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history, label='Raw Reward', color='blue')
    plt.title('Total Reward over Episodes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

