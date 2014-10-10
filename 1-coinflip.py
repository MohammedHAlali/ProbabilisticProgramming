import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

dist = stats.beta
x = np.linspace( 0, 1, 400 )

N = 800
data = stats.bernoulli.rvs( 0.5, size = N )
data

heads = data.sum()

y = dist.pdf( x, 1 + heads, 1 + N - heads )

plt.clf()
plt.plot( x, y )
plt.show(block=False)
plt.draw()
