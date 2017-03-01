import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.ion()
%load_ext autoreload
%autoreload 2

import bandit
from bandit import Bandit, MultiArmedBandit, Agent
mab = MultiArmedBandit(100)
a = Agent(1000, mab)
a.run_greedy()


fig = plt.figure(1)
fig.clear()
ax = fig.add_subplot(4, 1, 1)
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.plot( np.arange(a.num_steps), a.avg_reward, label='Avg reward', color='red')
ax.legend(loc='upper left')
ax = fig.add_subplot(4, 1, 2)
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.scatter( np.arange(a.num_steps), a.history_action, label='actions',
        color='black' )
ax.legend(loc='upper left')
ax = fig.add_subplot(4, 1, 3)
ax.plot( np.arange(a.num_steps), a.optimal_action, label='optimal action')
ax.legend(loc='upper left')
ax = fig.add_subplot(4, 1, 4)
ax.bar( np.arange(a.mab.num_bandits), a.count_action)
ax.set_xlabel("bandit")
fig.canvas.draw()
fig.show()


agents = [Agent(1000, MultiArmedBandit(10)) for b in range(2000)]

for a in agents:
    a.run_greedy()

avg_reward = np.array([a.avg_reward for a in agents])
avg_reward = avg_reward.mean(axis=0)

optimal_action = np.array([a.optimal_action for a in agents])
optimal_action = optimal_action.mean(axis=0)

fig = plt.figure(1)
fig.clear()
ax = fig.add_subplot(2, 1, 1)
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.plot( np.arange(a.num_steps), avg_reward, label=''  , color='red' )
ax.legend(loc='upper left')
ax = fig.add_subplot(2, 1, 2)
ax.set_ylim(0,1)
ax.plot( np.arange(a.num_steps), optimal_action)
fig.canvas.draw()
fig.show()


#
# optimistic initial condition vs q_0 = 0

for a in agents:
    a.optimistic_init()
    a.run_greedy()

optimal_action_optinit = np.array([a.optimal_action for a in agents])
optimal_action_optinit = optimal_action_optinit.mean(axis=0)


plt.tight_layout()
fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.plot( np.arange(a.num_steps), optimal_action, label='zero init', color='red')
ax.plot( np.arange(a.num_steps), optimal_action_optinit, label='+5 init',
        color='green')
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()

import multiprocessing

jobs = []
for a in agents:
    process = multiprocessing.Process(target=a.run_greedy)
    process.start()


def run_agent(id):
    a = Agent(1000, MultiArmedBandit(10))
    a.run_greedy()
    return a

with multiprocessing.Pool() as pool:
    ags = pool.map(run_agent, range(10))

ags[4].avg_reward


agents[0].avg_reward

