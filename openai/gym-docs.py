import gym

env = gym.make( "CartPole-v0" )

env.monitor.start('/tmp/cartpole-experiment-1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print( observation )
        action = env.action_space.sample() 
        observation, reward, done, info = env.step( action )
        print( reward )
        if done:
            print( "Episode {}".format(t+1) )
            break
env.monitor.close()

print( env.action_space )

print( env.observation_space )

print( env.observation_space.high )

print( env.observation_space.low )

space = gym.spaces.Discrete(8)

x = space.sample()

assert space.contains( x )

assert space.n == 8

for e in gym.envs.registry.all():
    print( e )


