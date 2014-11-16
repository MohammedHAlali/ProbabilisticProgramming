# Bayesian data analysis with PyMC3
# Thomas Wiecki
# Quantopian Inc.

# http://nbviewer.ipython.org/github/twiecki/pymc3_talk/blob/master/bayesian_pymc3.ipynb

import numpy  as np
import matplotlib.pyplot as plt
import prettyplotlib as ppl
from scipy import stats
from matplotlib import rc

# Coin flipping experiment
x_coin = np.linspace( 0, 1, 100 )

