import unittest
import numpy as np
import bandit
from bandit import Bandit, MultiArmedBandit, Agent

class TestBandit(unittest.TestCase):
    def test_Bandit(self):
        id = 3
        b = Bandit(id)
        self.assertIs(id, b.ID)

    def test_avg(self):
        mab = MultiArmedBandit(3)
        a = Agent(100, mab)
        a.run_greedy()
        avg = np.array(a.history_reward).mean()
        self.assertLess(avg - a.avg_reward[-1], 1e-6)

if __name__ == '__main__':
    unittest.main()
