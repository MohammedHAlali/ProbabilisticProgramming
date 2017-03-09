import numpy as np
import matplotlib.pyplot as plt
plt.ion()

%load_ext autoreload
%autoreload 2

from random_walk import RandomWalk

rw = RandomWalk()
for sim in rw.Simulate(3):
    print(sim)

from temporal_difference import *

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

# SARSA

