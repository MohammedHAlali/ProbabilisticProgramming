import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

%load_ext autoreload
%autoreload 2

from random_walk import RandomWalk
from temporal_difference import *

rw = RandomWalk()
for sim in rw.Simulate(3):
    print(sim)


s0 = 3
rw = RandomWalk()
value = np.zeros_like(rw.STATE, np.float) + 0.5
value[0] = 0.0
value[6] = 0.0
TD_zero_evaluation(rw, value, s0, 1000)
print(value)
plt.clf()
plt.plot(value, marker='o')
plt.ylim(0,1.1)

#
# SARSA

s0 = 3
rw = RandomWalk()
rw.Q_values[1:6,:] = 0.5
rw.policy = rw.eps_greedy_policy

rw.eps = 0.1
TD_zero_control(rw, s0, 10000)
plt.clf()
sns.heatmap(rw.Q_values, linewidths=1.5, square=True, annot=True)

[rw.greedy_policy(state) for state in rw.STATE]
