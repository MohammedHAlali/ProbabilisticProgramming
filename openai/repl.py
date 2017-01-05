import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

env = gym.make( "CartPole-v0" )

observation = env.reset()

def run_episode_df( env, parameters ):
    observation = env.reset()
    totalreward = 0
    df = pd.DataFrame( [observation] )
    for _ in xrange( 200 ):
        env.render()
        action = 0 if np.matmul( parameters , observation ) < 0 else 1
        observation, reward, done, info = env.step( action )
        df = df.append( [observation] , ignore_index=True )
        reward_centering = -1.1 * abs( observation[0] )
        totalreward += reward + reward_centering
        if done:
            break
    return totalreward , df

def run_episode( env, parameters):
    totalreward , df = run_episode_df( env, parameters )
    return totalreward


parameters = np.zeros(4)

totR , df = run_episode_df( env , parameters )

print(df)

fig = df.plot()
plt.show() 


fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()
