import numpy as np

from MDP import MDP


class RandomWalk(MDP):
    def __init__(self):
        super().__init__()
        self.STATE_NAMES = ['T0', 'A', 'B', 'C', 'D', 'E', 'T1']
        self.STATE = list(range(0, len(self.STATE_NAMES)))
        self.Q_values = np.zeros((len(self.STATE), 2))

    def TerminalState(self, state):
        return state == self.STATE[0] or state == self.STATE[-1]

    def random_policy(self, state):
        action = 2*np.random.randint(2)-1
        return action

    def greedy_policy(self, state):
        idx = np.argmax(self.Q_values[state, :])
        return 2*idx-1

    def SampleDynamics(self, state, action):
        s1 = state + action
        return s1

    def Reward(self, state):
        if state == self.STATE[-1]:
            return 1
        return 0

    def GetQ(self, state, action):
        a = int((action+1)/2)
        return self.Q_values[state, a]

    def SetQ(self, state, action, value):
        a = int((action+1)/2)
        self.Q_values[state, a] = value







