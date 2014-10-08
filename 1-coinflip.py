import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

N = 7000
data = stats.bernoulli.rvs( 0.5, size = N )
data

dist = stats.beta

x = np.linspace( 0, 1, 400 )


heads = data.sum()
heads = 3

y = dist.pdf( x, 1 + heads, 1 + N - heads )

plt.plot( x, y )
plt.show(block=False)
plt.draw()
