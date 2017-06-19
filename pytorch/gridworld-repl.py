# https://github.com/spro/practical-pytorch/blob/master/reinforce-gridworld/reinforce-gridworld.ipynb
# imports {{{
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
plt.ion()

%load_ext autoreload
%autoreload 2

from gridworld import *
# }}}

env = Environment()

env.reset()
done = False
while not done:
    _, _, done = env.step(2) # Down

anim = animate(env.history)
plt.gcf().canvas.manager.window.move(0,0)

anim.event_source.stop()


e = 0
while reward_avg < 0.75:
    actions, values, rewards = run_episode(e)
    final_reward = rewards[-1]
    discounted_rewards, value_loss = finish_episode(e, actions, values, rewards)
    reward_avg.add(final_reward)
    value_avg.add(value_loss.data[0])
    if e % log_every == 0:
        print('[epoch=%d]' % e, reward_avg, value_avg)
    if e > 0 and e % render_every == 0:
        animate(env.history)
    e += 1

# Plot average reward and value loss
plt.plot(np.array(reward_avg.avgs))
plt.show()
plt.plot(np.array(value_avg.avgs))
plt.show()


print(matplotlib.animation.writers.list())

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

plt.rcParams['animation.writer'] = 'ffmpeg'

plt.rcParams['animation.html']

anim.save('/tmp/movie.mp4')


multinomial
