import pymc as pm
import numpy  as np
import matplotlib.pyplot as plt

parameter = pm.Exponential( "poisson_param", 1 )
data_generator = pm.Poisson( "data_generator", parameter )
data_plus_one = data_generator + 1

# 'parents' influence another variable
# 'children' subject of parent vars
parameter.children
data_generator.parents
data_generator.children

# 'value' attribute
parameter.value
data_generator.value
data_plus_one.value

# 'stochastic' vars - still random even if parents are known
# 'deterministic' vars - not random if parents are known
# Initializing variables
#   * name argument - retrieves posterior dist 
#   * class specific arguments
#   * size - multivariate indp array of stochastic vars

some_var = pm.DiscreteUniform( "discrete_uni_var", 0, 4 )

betas = pm.Uniform( "betas", 0, 1, size=10 )
betas.value

# var.random() - generates new value
# var.value - returns new value
lambda_1 = pm.Exponential( "lambda_1", 1 )
lambda_2 = pm.Exponential( "lambda_2", 2 )
tau      = pm.DiscreteUniform( "tau", lower = 0, upper = 10 )

lambda_1.value
lambda_2.value
tau.value

lambda_1.random()
lambda_2.random()
tau.random()

lambda_1.value
lambda_2.value
tau.value

tau.value = 10

#
# deterministic var
n_data_points = 5  # in CH1 we had ~70 data points
@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_data_points)
    out[:tau] = lambda_1  # lambda before tau is lambda1
    out[tau:] = lambda_2  # lambda after tau is lambda2
    return out

type( lambda_1 + lambda_2 )

#
# observations
samples = [ lambda_1.random() for i in range(20000) ]
plt.hist( samples,
        bins = 70,
        normed = True,
        histtype = "stepfilled")
plt.xlim( 0, 8 )
plt.show()

# 'observed' - make value immutable

data = np.array( [10,5] )
fixed_variable = pm.Poisson( "fxd", 1, value = data, observed=True )
fixed_variable.value
fixed_variable.random()
fixed_variable.value

# fix txt msg to observed data set
data = np.array( [10, 20, 15, 20, 49] )
obs = pm.Poisson( "obs", lambda_, value=data, observed=True )
obs.value

# 
# Model class - analyze variables as a single unit
model = pm.Model( [obs, lambda_, lambda_1, lambda_2, taus] )

#
# Creating new datasets
#
maxdays = 80
tau = pm.rdiscrete_uniform( 0, maxdays )

alpha = 1 / 20.
lambda_1, lambda_2 = pm.rexponential( alpha, 2 )

data = np.r_[
        pm.rpoisson( lambda_1, tau ),
        pm.rpoisson( lambda_2, maxdays-tau )]

plt.bar( np.arange(maxdays), data )
plt.bar( tau - 1, data[tau-1], color = 'r', label='change point' )
plt.xlabel( "Time (days)" )
plt.ylabel( "count" )
plt.title( "Artificial Data" )
plt.xlim( 0, 80 )
plt.legend()
plt.show()

def plot_artificial_sms_dataset():
    maxdays = 80
    tau = pm.rdiscrete_uniform( 0, maxdays )
    alpha = 1 / 20.
    lambda_1, lambda_2 = pm.rexponential( alpha, 2 )
    data = np.r_[
            pm.rpoisson( lambda_1, tau ),
            pm.rpoisson( lambda_2, maxdays-tau )]
    plt.bar( np.arange(maxdays), data )
    plt.bar( tau - 1, data[tau-1], color = 'r', label='change point' )
    plt.xlim( 0, 80 )

for i in range(4):
    plt.subplot( 4, 1, i )
    plot_artificial_sms_dataset()
plt.show()

#
# A/B testing
#

#
# A only
p = pm.Uniform( 'p', lower=0, upper=1 )

p_true = 0.05
N = 1500
occurrences = pm.rbernoulli( p_true, N )
occurrences.sum()

print "observed frequency: %.4f" % occurrences.mean()

# run inference alg
obs = pm.Bernoulli( "obs", p, value=occurrences, observed=True )

mcmc = pm.MCMC( [p, obs] )
mcmc.sample( 18000, 1000 )

plt.title( 'Posterior Dist of $p_A$: the true effectiveness of site A' )
plt.vlines( p_true, 0, 90, linestyles="--", label="true $p_A$ (unknown)" )
plt.hist( mcmc.trace( "p" )[:], bins=25, histtype="stepfilled", normed=True )
plt.legend()
plt.show()

#
# A and B together

# infer p_A, p_B and delta = p_A - p_B
true_p_A = 0.05
true_p_B = 0.04
N_A = 1500
N_B = 750

observations_A = pm.rbernoulli( true_p_A, N_A )
observations_B = pm.rbernoulli( true_p_B, N_B )
print "Obs from Site A: ", observations_A[:30].astype(int), "..."
print "Obs from Site B: ", observations_B[:30].astype(int), "..."
print observations_A.mean()
print observations_B.mean()

p_A = pm.Uniform( "p_A", 0, 1 )
p_B = pm.Uniform( "p_B", 0, 1 )

@pm.deterministic
def delta( p_A=p_A, p_B=p_B ):
    return p_A - p_B

obs_A = pm.Bernoulli( "obs_A", p_A, value=observations_A, observed=True )
obs_B = pm.Bernoulli( "obs_B", p_B, value=observations_B, observed=True )

mcmc = pm.MCMC( [p_A, p_B, delta, obs_A, obs_B] )
mcmc.sample( 20000, 1000 )

p_A_samples = mcmc.trace( "p_A" )[:]
p_B_samples = mcmc.trace( "p_B" )[:]
delta_samples = mcmc.trace( "delta" )[:]

ax = plt.subplot( 311 )
plt.xlim(0, .1)
plt.hist(p_A_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_A$", color="#A60628", normed=True)
plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
plt.legend(loc="upper right")
plt.title("Posterior distributions of $p_A$, $p_B$, and delta unknowns")
ax = plt.subplot(312)
plt.xlim(0, .1)
plt.hist(p_B_samples, histtype='stepfilled', bins=25, alpha=0.85,
         label="posterior of $p_B$", color="#467821", normed=True)
plt.vlines(true_p_B, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
plt.legend(loc="upper right")
ax = plt.subplot(313)
plt.hist(delta_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of delta", color="#7A68A6", normed=True)
plt.vlines(true_p_A - true_p_B, 0, 60, linestyle="--",
           label="true delta (unknown)")
plt.vlines(0, 0, 60, color="black", alpha=0.2)
plt.legend(loc="upper right");
plt.show()

# 
print "Prob site A is WORSE  than B: %.3f" % (delta_samples < 0).mean()
print "Prob site A is BETTER than B: %.3f" % (delta_samples > 0).mean()




