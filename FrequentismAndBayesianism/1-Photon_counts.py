# Frequentism and Bayesianism: A Practical Introduction
# http://jakevdp.github.io/blog/2014/03/11/frequentism-and-bayesianism-a-practical-intro/

# Frequentist - limiting case of repeate measuremnts
# Bayesians - degree of uncertainty about statements

# Simple photon counts
# Assume star's true flux is constant in time, fixed at F_true
# N measuremnts
# D = { F_i, e_i } : observed photon flux F_i and error e_i

# Generate simple photon counts
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(1)
F_true = 1000
N = 50
F = stats.poisson( F_true ).rvs(N)
e = np.sqrt( F )

# viz measured data {{{
fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("Flux")
ax.set_ylabel("measuremnts")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.errorbar( F, np.arange(N)
        , xerr = e
        , fmt = 'ok'
        , ecolor = 'gray'
        , alpha = 0.5
        )
ax.vlines( F_true, 0, N
        , linewidth=5
        , alpha = 0.2
        )
fig.canvas.draw()
fig.show()
# }}}

# Frequentist approach - Maximum Likelihood
# P[ D_i | F_true ] = N( F_true, e_i )[F_i]
# max L( D_i | L_true ) = Prod( P[ D_i | F_true ], i )
# max log L() = max L()

w = 1. / (e**2)
print("""
        F_true = {0}
        F_est  = {1:.0f} +/- {2:.0f} (based on {3} measuremnts)
""".format(
    F_true
    , (w * F).sum() / w.sum()
    , w.sum() ** -0.5
    , N
    )
)


# Bayesian approach
# Want P[ F_true | D ]
# P[ F_true | D ] = P[ D | F_true ] * P[ F_true ] / P[ D ]
#   P[ F_true | D ]  -  the posterior ( what we want )
#   P[ D | F_true ]  -  The likelihood ( similar to frequentist approach )
#   P[ F_true ]      -  the prior ( what we knew )
#   P[ D ]           -  the data prob ( simply a normalization term )

import pymc as pm

F_prior = pm.Uniform( "prior", 0, 2000 )
F_obs = pm.Poisson( "photon_counts", mu=F_prior, observed=True, value=F )

model = pm.Model( [
    F_prior,
    F_obs
    ] )

mcmc = pm.MCMC( model )
mcmc.sample( iter=50*2000, burn=1000 )

F_trace = mcmc.trace( "prior" )[:]

F_fit = np.linspace( 975, 1025 )
pdf = stats.norm( 
          np.mean( F_trace )
        , np.std( F_trace )
        ).pdf( F_fit )

fig = plt.figure()
fig.clear()
ax = fig.gca()
ax.set_xlabel("F")
ax.set_ylabel("P(F)")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.hist( F_trace
        , bins = 50
        , histtype = "stepfilled"
        , alpha = 0.3
        , normed = True
        )
ax.plot( F_fit, pdf )
fig.canvas.draw()
fig.show()

print( """
        F_true = {0}
        F_est  = {1:.0f} +/- {2:.0f} (based on {3} measuremnts)
""".format(
    F_true
    , np.mean( F_trace )
    , np.std( F_trace )
    , N
    ) )
