import gym
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
plt.ion()


env = gym.make( "CartPole-v0" )

# rand [-1 , 1]
parameters = 2*np.random.rand(4) - 1


def run_episode( env, parameters ):
    observation = env.reset()
    totalreward = 0
    for _ in xrange( 200 ):
        env.render()
        action = 0 if np.matmul( parameters , observation ) < 0 else 1
        observation, reward, done, info = env.step( action )
        totalreward += reward
        if done:
            break
    return totalreward

run_episode( env, parameters )

bestParam = None
bestReward = 0
for _ in xrange(100):
    parameters = 2*np.random.rand(4) - 1
    reward = run_episode( env , parameters )
    if reward > bestReward:
        bestReward = reward
        bestParam = parameters
        if reward >= 200:
            break

run_episode( env, bestParam )

noise_scaling = 0.01
# parameters = np.random.rand(4) * 2 - 1
# parameters = np.zeros(4)
bestReward = 0
for _ in xrange( 100 ):
    dp = (2*np.random.rand(4)-1) * noise_scaling
    newparams =  parameters + dp
    reward = run_episode( env, newparams )
    if reward > bestReward:
        bestReward = reward
        parameters = newparams
        if reward >= 200:
            print( "SOLUTION FOUND" )
            break

def policy_gradient():
    params = tf.get_variable( "policy_parameters" , [4,2] )
    state = tf.placeholder( "float", [None,4] )
    linear = tf.matmul( state, params )
    probabilities = tf.nn.softmax( linear )
    good_probabilities = tf.reduce_sum( tf.mul( probabilities , actions ) 
            , reduction_indices=[1] )
    log_probabilities = tf.log( good_probabilities )
    loss = -tf.reduce_sum( log_probabilities )
    optimizer = tf.train.AdamOptimizer( 0.1 ).minimize( loss )


def value_gradient():  
    # sess.run(calculated) to calculate value of state
    state = tf.placeholder("float",[None,4])
    w1 = tf.get_variable("w1",[4,10])
    b1 = tf.get_variable("b1",[10])
    h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
    w2 = tf.get_variable("w2",[10,1])
    b2 = tf.get_variable("b2",[1])
    calculated = tf.matmul(h1,w2) + b2
    # sess.run(optimizer) to update the value of a state
    newvals = tf.placeholder("float",[None,1])
    diffs = calculated - newvals
    loss = tf.nn.l2_loss(diffs)
    optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)


# tensorflow operations to compute probabilties for each action, given a state
pl_probabilities, pl_state = policy_gradient()  
observation = env.reset()  
actions = []  
transitions = []  
for _ in xrange(200):  
    # calculate policy
    obs_vector = np.expand_dims(observation, axis=0)
    probs = sess.run(pl_probabilities,feed_dict={pl_state: obs_vector})
    action = 0 if random.uniform(0,1) < probs[0][0] else 1
    # record the transition
    states.append(observation)
    actionblank = np.zeros(2)
    actionblank[action] = 1
    actions.append(actionblank)
    # take the action in the environment
    old_observation = observation
    observation, reward, done, info = env.step(action)
    transitions.append((old_observation, action, reward))
    totalreward += reward
    if done:
        break
