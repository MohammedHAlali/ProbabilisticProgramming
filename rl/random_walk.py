import numpy as np

from MDP import MDP


class RandomWalk(MDP):
    def __init__(self):
        super().__init__()
        self.STATE_NAMES = ['T0', 'A', 'B', 'C', 'D', 'E', 'T1']
        self.STATE = list(range(0, len(self.STATE_NAMES)))

    def TerminalState(self, state):
        return state == self.STATE[0] or state == self.STATE[-1]

    def policy(self, state):
        action = 2*np.random.randint(2)-1
        return action

    def SampleDynamics(self, state, action):
        s1 = state + action
        return s1

    def Reward(self, state):
        if state == self.STATE[-1]:
            return 1
        return 0







