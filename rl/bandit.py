import numpy as np

def inc_avg(x, avg, N):
    return avg + (x - avg)/N

class Bandit(object):
    def __init__(self, id):
        self.expected_reward = np.random.randn()
        self.ID = id

    def __str__(self):
        return "{}, reward: {}".format(self.ID, self.expected_reward)

    def Reward(self):
        return np.random.randn() + self.expected_reward

class MultiArmedBandit():
    def __init__(self, num_bandits):
        self.num_bandits = num_bandits
        self.bandits = [Bandit(i) for i in range(num_bandits)]

    def __str__(self):
        return '\n'.join(["{}".format(b) for b in self.bandits])

    def Reward(self, action):
        return self.bandits[action].Reward()

    def OptimalBandit(self):
        return np.argmax([b.expected_reward for b in self.bandits])

class Agent():
    def __init__(self, num_steps, mab):
        self.num_steps = num_steps
        self.avg_reward = np.zeros(num_steps)
        self.optimal_action = np.zeros(num_steps)
        self.Q_values = np.zeros(mab.num_bandits)
        self.mab = mab
        self.history_action = []
        self.history_reward = []
        self.count_action = np.zeros(mab.num_bandits, dtype=np.int)

    def greedy(self):
        return np.argmax(self.Q_values)

    def eps_greedy(self):
        eps = 1e-1
        if np.random.rand() < eps:
            return np.random.randint(self.mab.num_bandits)
        else:
            return self.greedy()

    def run_greedy(self):
        optimal_bandit = self.mab.OptimalBandit()
        for t in range(self.num_steps):
            action = self.eps_greedy()
            reward = self.mab.Reward(action)
            self.count_action[action] += 1
            self.avg_reward[t] = inc_avg(reward, self.avg_reward[t-1], t+1)
            self.Q_values[action] = inc_avg(reward, self.Q_values[action],
                    self.count_action[action])
            self.history_action.append(action)
            self.history_reward.append(reward)
            opt = 0
            if action == optimal_bandit:
                opt = 1
            self.optimal_action[t] = inc_avg( opt, self.optimal_action[t-1]
                    , t+1)

    def optimistic_init(self):
        self.Q_values = np.repeat(5.0, self.mab.num_bandits)


