import numpy  as np
import matplotlib.pyplot as plt
import pymc as pm

count_data = np.loadtxt( "/home/abergman/apps/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/Chapter1_Introduction/data/txtdata.csv" )

n_count_data = len( count_data )

plt.bar( np.arange(n_count_data), count_data )
plt.xlabel('Time (days)')
plt.ylabel( 'count of txt-msg recv' )
plt.title( 'Texting habit' )
plt.xlim( 0, n_count_data )
plt.show()

alpha = 1.0 / count_data.mean()

lambda_1 = pm.Exponential( "lambda_1", alpha )
lambda_2 = pm.Exponential( "lambda_2", alpha )

tau = pm.DiscreteUniform( "tau", 
        lower = 0, 
        upper=n_count_data)

print lambda_1.random()

x = np.array( [lambda_1.random() for i in xrange(100000)] )

plt.hist( x, bins = 400 )
plt.show()

@pm.deterministic
def lambda_( tau=tau, lambda_1=lambda_1, lambda_2=lambda_2 ):
    out = np.zeros( n_count_data )
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

observation = pm.Poisson( "obs", lambda_, value = count_data, observed=True )

model = pm.Model( [observation, lambda_1, lambda_2, tau] )

mcmc = pm.MCMC( model )

mcmc.sample( 40000, 1000, 1 )

lambda_1_samples = mcmc.trace( 'lambda_1' )[:]
lambda_2_samples = mcmc.trace( 'lambda_2' )[:]
tau_samples = mcmc.trace( 'tau' )[:]


ax = plt.subplot( 311 )
ax.set_autoscaley_on( False )

plt.hist( lambda_1_samples,
        histtype = 'stepfilled',
        bins = 30,
        alpha = 0.85,
        label = "posterior of $lambda_1$",
        color="#A60628",
        normed=True)

plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data) - 20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability");

plt.show()


N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range( 0, n_count_data ):
    ix = day < tau_samples
    expected_texts_per_day[ day ] = (lambda_1_samples[ix].sum() +
                                     lambda_2_samples[~ix].sum() ) / N


plt.plot( range(n_count_data), expected_texts_per_day,
        lw=4,
        color="#E24A33",
        label="expected num txt msg recv")
plt.xlim( 0, n_count_data )
plt.ylim( 0, 60 )
plt.xlabel( "Day" )
plt.ylabel( "Exp # txt msg" )
plt.title( "Exp num txt msg recv" )
plt.bar( np.arange( len(count_data) ),
        count_data,
        color = "#348ABD",
        alpha = 0.65,
        label = "observed texts per day")
plt.legend(loc="upper left")
plt.show()

# 1. mean 
lambda_1_mean = lambda_1_samples.mean()
lambda_2_mean = lambda_2_samples.mean()

# 2. inc in txt msg
lambda_1_div_2 = lambda_1_samples / lambda_2_samples
lambda_1_div_2.mean() 

lambda_1_mean / lambda_2_mean

#3. E[ lambda_1 | tau < 45 ]
ix = tau_samples < 45

lambda_1_samples[ix].mean()

