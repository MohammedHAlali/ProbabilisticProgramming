import numpy  as np
import matplotlib.pyplot as plt
import pymc as pm

# P( "YES" ) = P( fc=HEADS )*P( cheater ) + P( fc=TAILS )*P( sc=HEADS )
#            = 0.5 * p + 0.5 * 0.5
p = pm.Uniform( "freq_cheating", 0, 1 )

@pm.deterministic
def p_skewed( p=p ):
    return 0.5 * p + 0.25

# P( "YES" ) = p_skewed
# Binomial( N, p_skewed ) := # of yes responses
# we observed 35 "yes" responses in our experiment
yes_responses = pm.Binomial( "number_cheaters", 100, p_skewed
        , value = 35
        , observed = True 
        )

# Model container and black box mcmc alg
model = pm.Model( [yes_responses, p_skewed, p])
mcmc = pm.MCMC( model )
mcmc.sample( 25000, 2500 )

p_trace = mcmc.trace( "freq_cheating" )[:]
fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(0,1)
ax.hist( p_trace
        , histtype="stepfilled"
        , label='posterior dist'
        , color='#348ABD'
        , normed=True
        , alpha=0.85
        , bins=30
        )
ax.legend(loc='upper right')
fig.canvas.draw()
fig.show()


