import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

%load_ext autoreload
%autoreload 2

import value_iteration as vi

vi.STATES

vi.ALL_STATES

vi.ACTIONS[76]

vi.NextStates(76,0)

[vi.ProbabilityTransition(s1,33,3) for s1 in vi.NextStates( 33, 3)]

[ vi.ProbabilityTransition(30,33,3), 
  vi.ProbabilityTransition(36,33,3) ]

vi.ProbabilityTransition(100,100,0)

vi.Reward(10, 1)

value = np.zeros(101)
vi.ExpectedReturn( 100, 0, value)

value = np.zeros(101)
value1, iter = vi.ValueIteration( value )
print(iter)
plt.clf()
plt.title('V(s)')
plt.xlabel('capital')
plt.ylabel('probability MAXCAP')
plt.scatter(vi.ALL_STATES, value1)
np.max(value1)

policy = vi.GetPolicy( value1 )
plt.clf()
plt.title('Policy')
plt.xlabel('capital')
plt.ylabel('stake')
plt.scatter( vi.ALL_STATES, policy)

r = [p/s for s, p in enumerate(policy)]
plt.clf()
plt.plot(r, marker='o')
plt.xlabel('capital')
plt.ylabel('stake/capital')

Qvalue = vi.GetQValue( value1 )
Qvalue_mat = np.zeros((55,101))
for s, actions in enumerate(Qvalue):
    for a, q in enumerate(actions):
        Qvalue_mat[a,s] = q
plt.clf()
sns.heatmap( Qvalue_mat, cmap='copper', square=True, mask=Qvalue_mat>1.0 )
plt.xlabel('capital')
plt.ylabel('stake')
plt.title('Q values')
np.max(Qvalue_mat)

plt.clf()
plt.plot( Qvalue[76], marker='o' )


np.argmax(Qvalue_mat)

Qvalue[100]

s0 = 20
histories = []
for _ in range(300):
    histories.append(vi.Simulate( s0, policy))
fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("rollouts")
ax.set_xlabel("time")
ax.set_ylabel("capital")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
for h in histories:
    ax.plot(h,marker='o')
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()

plt.clf()
plt.hist([h[-1] for h in histories], normed=True)


