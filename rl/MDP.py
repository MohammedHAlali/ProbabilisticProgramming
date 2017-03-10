import numpy as np


class MDP(object):
    def __init__(self):
        self.discount = 1
        self.states = []
        self.actions = []
        self.MAX_ITER = 10000
        self.eps = 1e-1

    def SampleDynamics(self, state, action):
        return state

    def Reward(self, state):
        return None

    def TerminalState(self, state):
        return True

    def policy(self, state):
        return None

    def eps_greedy_policy(self, state):
        if np.random.rand() < self.eps:
            return self.random_policy(state)
        else:
            return self.greedy_policy(state)

    def Simulate_History(self, s0):
        history = []
        for iter in range(self.MAX_ITER):
            action = self.policy(s0)
            s1 = self.SampleDynamics(s0, action)
            reward = self.Reward(s1)
            history.append((s0, action, reward))
            if self.TerminalState(s1):
                history.append((s1, None, None))
                break
            s0 = s1
        return history

    def Simulate(self, s0):
        a0 = self.policy(s0)
        for iter in range(self.MAX_ITER):
            s1 = self.SampleDynamics(s0, a0)
            reward = self.Reward(s1)
            a1 = self.policy(s1)
            yield (s0, a0, reward, s1, a1)
            if self.TerminalState(s1):
                # yield (s1, None, None, s1)
                return
            s0 = s1
            a0 = a1
        return
