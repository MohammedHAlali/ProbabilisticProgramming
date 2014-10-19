import numpy  as np
import matplotlib.pyplot as plt
import pymc as mc
import scipy.stats as stats
import math

# http://blog.yhathq.com/posts/estimating-user-lifetimes-with-pymc.html

# artificial data
N = 20
true_alpha = 2
true_beta = 5
lifetime = mc.rweibull( true_alpha, true_beta, size=N )
birth = mc.runiform( 0, 10, N )

# an individual is right censored if this is true
censor = (birth + lifetime) > 10
lifetime_ = np.ma.masked_array( lifetime, censor )
lifetime_.set_fill_value( 10 )

plt.clf()
y = np.arange( 0, N )
for b,l,yy in zip( birth, lifetime, y ):
    plt.plot( [b,b+l], [yy,yy] )
plt.plot( birth+lifetime, y, linestyle="", marker="o" )
plt.draw()
plt.show( block=False )

# begin the model
# just use uniform priors
alpha = mc.Uniform( "alpha", 0, 20 )
beta  = mc.Uniform( "beta", 0, 20 )
obs   = mc.Weibull( "obs", alpha, beta, value = lifetime_, observed = True )

#
# Censor factor
# if any guess doesnt follow the hard limit that some samples are
# right-censored, then we discard the guess and try again
@mc.potential
def censor_factor( obs=obs ):
    if np.any( (obs+birth < 10)[lifetime_.mask] ):
        return -100000
    else:
        return 0

# perform MCMC
mcmc = mc.MCMC( [alpha, beta, obs, censor_factor] )
mcmc.sample( 50000, 30000 )

# Posterior dist
plt.clf()
for x,c in zip( ['alpha', 'beta'], ["#348ABD", "#A60628"] ):
    plt.hist( mcmc.trace( x )[:]
            , histtype = 'stepfilled'
            , bins = 30
            , alpha = 0.5
            , label = "posterior of $\\" + x + "$" 
            , color=c
            , normed=True
            )
plt.vlines( true_alpha, 0, 1.0, linestyle="--", label=r"True $\alpha$" )
plt.vlines( true_beta,  0, 1.0, linestyle="--", label=r"True $\beta$" )
plt.legend( )
plt.xlim( 0, 14 )
plt.title( 'Posterior distributions of $\\alpha, \\beta$' )
plt.draw()
plt.show( block=False )

# median lifetime
samples_alpha =mcmc.trace( 'alpha' )[:] 
samples_beta = mcmc.trace( 'beta'  )[:]

# Weibull pdfs
x = np.linspace( 0, 10, 100 )
plt.clf()
for a,b in zip( samples_alpha, samples_beta )[1:1000]:
    w = stats.weibull_min( c = a, scale = b ).pdf(x)
    plt.plot( x, w, alpha = 0.1, color = "lavender" )
w = stats.weibull_min( c = 2, scale = 5 ).pdf(x)
plt.plot( x, w )
plt.draw()
plt.show( block=False )


median = samples_beta * (np.log( 2 ))**(1/samples_alpha)

plt.clf()
plt.hist( median
        , bins=30
        , color="lavender"
        , normed=True
        )
plt.title( "Posterior median lifetime" )
plt.xlim( 1,7 )
plt.draw()
plt.show( block=False )


# more data points
N = 2500
lifetime = mc.rweibull( true_alpha, true_beta, size=N )
birth = mc.runiform( 0, 10, N )

# an individual is right-censored if this is True
censor = ((birth+lifetime) >= 10)
lifetime_ = lifetime.copy()
lifetime_[censor] = 10 - birth[censor]

# uniform priors
alpha = mc.Uniform( "alpha", 0, 20 )
beta  = mc.Uniform( "beta", 0, 20 )

@mc.observed
def survival( value=lifetime_, alpha = alpha, beta = beta ):
    return sum( (1-censor)(math.log( alpha/beta) + (alpha-1)*np.log(value/beta)) - (value/beta)**(alpha) )

@mc.observed
def survival( value=lifetime_, alpha = alpha, beta = beta ):
    return sum((1-censor)(math.log( alpha/beta) + (alpha-1)*np.log(value/beta)))

mcmc = mc.MCMC( [alpha, beta, survival] )
mcmc.sample( 50000, 30000 )

