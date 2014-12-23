import numpy  as np
import matplotlib.pyplot as plt
import pymc as pm
import scipy.stats as stats
from scipy.stats.mstats import mquantiles
from separation_plot import separation_plot

# Protip: Lighter deterministic vars with Lambda class
#   - elementary math operations can produce deterministic vars implicitly
#   - indexing or slicing operations need Lambda functions
N=10
beta = pm.Normal( "coefficients", mu=0, tau=1, size=(N,1) )
x = np.random.randn( N, 1 )
linear_combination = pm.Lambda( "mylambda"
        , lambda x = x, beta=beta: np.dot( x.T, beta ) )


# Protip: Arrays of PyMC variables
# store multiple heterogenous PyMC variables in a Numpy Array
N=10
x = np.empty( N, dtype=object )
for i in range( 0, N ):
    x[i] = pm.Exponential( 'x_%i' % i, (i+1)**2 )
print x

#
# Example: Challenger Space Shuttle Disaster
np.set_printoptions( precision=3, suppress=True )
challenger_data = np.genfromtxt( 
"/home/abergman/apps/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/Chapter2_MorePyMC/data/challenger_data.csv"
        , skip_header = 1
        , usecols=[1,2]
        , missing_values="NA"
        , delimiter="," )
challenger_data = challenger_data[ ~np.isnan( challenger_data[:,1] ) ]

fig = plt.figure()
fig.clear()
ax = fig.gca()
ax.set_title("Defects vs temp")
ax.set_xlabel("Outside temp (F)")
ax.set_ylabel("Damage Incident?")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.set_yticks( [0,1] )
ax.scatter( challenger_data[:,0], challenger_data[:,1]
        , label=''
        , color='k' 
        , s=75
        , alpha=0.5)
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()

# Q: At Temp t, what is the probability of a damage incident?

# need a function bounded between 0 and 1
# Popular choice is the logistic function:
def logistic( x, beta, alpha=0 ):
    return 1.0 / ( 1.0 + np.exp( np.dot( beta, x ) + alpha ) )

x = np.linspace( -4, 4, 100 )
fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.plot( x , logistic( x , 1 )  , label=r'$\beta = 1$'  , color='red',  lw=1   , ls="--" )
ax.plot( x , logistic( x , 3 )  , label=r'$\beta = 3$'  , color='blue', lw=1      , ls="--" )
ax.plot( x , logistic( x , -5 ) , label=r'$\beta = -5$' , color='green', lw=1     , ls="--" )
ax.plot( x , logistic( x , 1  , 1 )  , label=r'$\beta = 1  , \alpha=1$'  , color='red' , lw=2)
ax.plot( x , logistic( x , 3  , -2 ) , label=r'$\beta = 3  , \alpha=-2$' , color='blue', lw=2 )
ax.plot( x , logistic( x , -5 , 7 )  , label=r'$\beta = -5 , \alpha=7$'  , color='green', lw=2 )
ax.legend(loc='upper right')
fig.canvas.draw()
fig.show()

# The Normal Distribution 
# alpha and beta have no reason to be positive / bounded / or relatively large
# modeled as a Normal rv
# X ~ N( mu, 1 / tau )
#   mu: mean
#   tau: precision,   1 / tau = sigma^2

nor = stats.norm
x = np.linspace( -8, 7, 300 )
mu = (-2, 0, 3)
tau = ( 0.7, 1, 2.8 )
colors = ["#348ABD", "#A60628", "#7A68A6"]
parameters = zip( mu, tau, colors )

for _mu, _tau, _color in parameters:
    plt.plot( x, nor.pdf(x, _mu, scale=1./_tau )
        , color = _color
        , label = "$\mu = %d,\;\\tau = %.1f$" % (_mu, _tau)
        )
    plt.fill_between( x, nor.pdf( x, _mu, scale=1./_tau )
            , color = _color
            , alpha = 0.33
            )
plt.legend( loc="upper right" ) 
plt.xlabel( "$x$" )
plt.ylabel( "density function at $x$" )
plt.title( "prob dist" )
plt.show()


# 
# Challenger space craft
temperature = challenger_data[:,0]
# Defect or not?
D = challenger_data[:, 1]

# value set to zero since if alpha/beta very large, p = 1 or 0, unstable
# no effect on results, or prior. Just a computational caveat
beta = pm.Normal( "beta", 0, 0.001, value=0 )
alpha = pm.Normal( "alpha", 0, 0.001, value=0 )

@pm.deterministic
def p( t=temperature, alpha=alpha, beta=beta ):
    return 1.0 / ( 1.0 + np.exp( beta * t + alpha ) )

print p.value

# connect probabilities in 'p' with our observations
observed = pm.Bernoulli( "bernoulli_obs", p, value=D, observed=True )

# train model on observed data
model = pm.Model( [observed, beta, alpha] )
map_ = pm.MAP( model )
map_.fit()
mcmc = pm.MCMC( model )
mcmc.sample( 120000, 100000, 2 )

alpha_samples = mcmc.trace( 'alpha' )[:,None]
beta_samples = mcmc.trace( 'beta' )[:,None]

plt.subplot( 211 )
plt.title( r"Post dist of $\alpha, \beta$" )
plt.hist( beta_samples
        , histtype="stepfilled"
        , bins=35
        , alpha=0.85
        , label=r"post of $\beta$"
        , color="#7A68A6"
        , normed=True
        )
plt.legend()
plt.subplot( 212 )
plt.hist( alpha_samples
        , histtype="stepfilled"
        , bins=35
        , alpha=0.85
        , label=r"post of $\alpha$"
        , color="#A60628"
        , normed=True
        )
plt.legend()
plt.show()

#
# expected probability - avg over all samples from posterior to get likely p(t_i)
t = np.linspace( temperature.min() - 5, temperature.max() + 5, 50 )[:,None]
p_t = logistic( t.T, beta_samples, alpha_samples )
mean_prob_t = p_t.mean( axis = 0 )

plt.plot( t, mean_prob_t
        , lw=3
        , label="avg posterior \nprob of defect"
        )
plt.plot( t, p_t[0,:]
        , ls="--"
        , label="realization from posterior"
        )
plt.plot( t, p_t[-2,:]
        , ls="--"
        , label="realization from posterior"
        )
plt.scatter( temperature, D
        , color="k"
        , s=50
        , alpha=0.5
        )
plt.title( "Posterior expected value of prob of defect; \
        plus realization" )
plt.legend( loc="lower left" )
plt.ylim( -0.1, 1.1 )
plt.xlim( t.min(), t.max() )
plt.ylabel( "prob" )
plt.xlabel( "temp" )
plt.show()

# Expected value and 95% interval for each temp
qs = mquantiles( p_t, [0.025, 0.975], axis=0 )
plt.fill_between( t[:,0], *qs, alpha=0.7, color="#7A68A6" )
plt.plot( t[:,0], qs[0]
        , label="95% CI" 
        , color = "#7A68A6"
        , alpha=0.7 
        )
plt.plot( t, mean_prob_t
        , lw=1
        , ls="--"
        , color="k"
        , label="avg posterior \nprob of defect"
        )
plt.xlim( t.min(), t.max() )
plt.ylim( -0.02, 1.02 )
plt.legend( loc="lower left" )
plt.scatter( temperature, D
        , color="k"
        , s=50
        , alpha=0.5
        )
plt.xlabel( "temp $t$" )
plt.ylabel( "prob estimate" )
plt.title( "Posterior prob estimate given temp $t$" )
plt.show()

# P( t = 31 ) 
# on the day of the challenger disaster, the outside temp was 31F
prob_31 = logistic( 31, beta_samples, alpha_samples )
fig = plt.figure()
fig.clear()
ax = fig.gca()
ax.set_title("Poster dist of prob defect, give $t=31$")
ax.set_xlabel("prob of defect occuring in O-ring")
ax.set_ylabel("")
ax.set_xlim( 0.995, 1.0 )
ax.set_ylim(auto=True)
ax.hist( prob_31
        , bins = 1000
        , normed = True
        , histtype = "stepfilled"
        )
fig.canvas.draw()
fig.show()

#
# Goodness of fit
# how can we test if our model is a bad fit?
# sample from the posterior dist and compare artificial dataset to the observed dataset
simulated = pm.Bernoulli( "bernoulli_sim", p )
N = 10000
mcmc = pm.MCMC( [simulated, alpha, beta, observed])
mcmc.sample(N)

simulations = mcmc.trace("bernoulli_sim")[:]
print simulations.shape

plt.title( "Simulated dataset using posterior parameters" )
for i in range(4):
    ax = plt.subplot( 4, 1, i+1)
    plt.scatter( temperature, simulations[1000*i,:]
            , color='k'
            , s=50
            , alpha=0.6 
            )
plt.show()

# separation plots - a graphical test of goodness of fit
# Bayesian p-values - another test
posterior_probability = simulations.mean(axis=0)
ix = np.argsort( posterior_probability )
print "posterior prob of defect | realized defect "
for i in range( len(D) ):
    print "%0.2f                     |  %d" % (posterior_probability[ix[i]], D[ix[i]])

separation_plot()







