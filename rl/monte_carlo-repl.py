import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

%load_ext autoreload
%autoreload 2

import monte_carlo as mc
from bandit import inc_avg

def plot_policy(policy):
    plt.clf()
    plt.title('Policy')
    plt.xlabel('capital')
    plt.ylabel('stake')
    plt.scatter(mc.ALL_STATES, policy)

def plot_qvalues(Q_values):
    plt.clf()
    sns.heatmap( Q_values, cmap='copper', square=True)
    plt.xlabel('stake')
    plt.ylabel('capital')
    plt.title('Q values')

def plot_v(v):
    plt.clf()
    plt.title('V(s)')
    plt.xlabel('capital')
    plt.ylabel('probability MAXCAP')
    plt.scatter(mc.ALL_STATES, v)

value = np.zeros(101)
value1, iter = mc.ValueIteration( value )
print("Iter: ", iter)
policy = mc.GetPolicy( value1 )

policy = mc.init_policy()

s0 = 90
hist, G = mc.SampleTrajectory(s0, policy)
print(hist, ' ', G)


v = mc.MonteCarloEvaluation( 51, policy, 1)
plt.clf()
plt.title('V(s)')
plt.xlabel('capital')
plt.ylabel('probability MAXCAP')
# plt.scatter(mc.ALL_STATES, value1, label='Value Iteration')
plt.scatter(mc.ALL_STATES, v, label='Monte Carlo Evaluation')

policy = mc.init_policy()
hist, G = mc.SampleTrajectory_ES( 30, 3, policy)
print(hist[0:20])

Q_values = mc.init_Q() + 1.0
policy = mc.init_policy()

policy = mc.MonteCarloControl_ES( Q_values, policy, 10000 )
plot_policy(policy)
plot_qvalues(Q_values)

v = [Q_values[s, policy[s]] for s in mc.ALL_STATES]
plot_v(v)

v2 = [Q_values[s, policy[s]] for s in mc.ALL_STATES]
plt.clf()
plt.title('V(s)')
plt.xlabel('capital')
plt.ylabel('probability MAXCAP')
plt.scatter(mc.ALL_STATES, v)
plt.scatter(mc.ALL_STATES, v2)
