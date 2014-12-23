# The Binomial Distribution
# X ~ Bin( N, p )
# params: N - num trials
#         p - prob in single trial
# E[ X ] = N*p

import numpy  as np
import matplotlib.pyplot as plt
import scipy.stats as stats

binomial = stats.binom
parameter = [(10, 0.4), (10,.9)]
colors = ["#348ABD", "#A60628"]

for i in range(2):
    N, p = parameter[i]
    _x = np.arange( N+1 )
    plt.bar( _x - 0.5, binomial.pmf(_x, N, p), 
            color=colors[i],
            edgecolor=colors[i],
            alpha=0.5,
            label="$N$: %d, $p$: %.1f" % (N,p),
            linewidth=3
            )

plt.legend( loc="upper left" )
plt.xlim( 0, 10.5 )
plt.xlabel( "$k$" )
plt.ylabel( "$P( X = k )$" )
plt.title( "Prob" )
plt.show()

#
# cheating during exam
# N: total number of students who took the exam
# X: "yes, i did cheat"
# p: proportion of cheaters
#
# The naive model:
# find posterior dist 'p'
# given 'N', prior on 'p' and observed data 'X'
# HOWEVER, students are not honest
#
# The privacy algorithm: 
# flip a coin
#   HEADS: answer honestly
#   TAILS: flip again
#     HEADS: "YES I CHEATED"
#     TAILS: "NO, I DID NOT"

import pymc as pm

# interview 100 students
N=100
# prior on 'p' uniform since we are completely ignorant
p = pm.Uniform( "freq_cheating", 0, 1 )
# data generation
true_answers = pm.Bernoulli( "truths", p, size=N )
# model first flip of Privacy Alg
first_coin_flip = pm.Bernoulli( "first_flip", 0.5, size=N )
print first_coin_flip.value
# model second flip
second_coin_flip = pm.Bernoulli( "second_flip", 0.5, size=N )

# observed proportion of yes responses
# 1 iff i) fc=HEADS and student cheated or
#      ii) fc=TAILS and sc=HEADS
# 0 o/w
@pm.deterministic
def observed_proportion( 
        t_a = true_answers,
        fc = first_coin_flip,
        sc = second_coin_flip):
    observed = fc * t_a + (1-fc) * sc
    return observed.sum() / float(N)

print observed_proportion.value

# no cheaters: expect to see 1/4 of all responses being yes
# everbody cheats: expect to see 3/4 of all responses being yes

# researcher recieve 35 "Yes" responses
X = 35

observations = pm.Binomial( "obs", N, observed_proportion, observed=True, value=X )

model = pm.Model( [
    p, 
    true_answers, 
    first_coin_flip, 
    second_coin_flip,
    observed_proportion, 
    observations] )

mcmc = pm.MCMC( model )
mcmc.sample( 40000, 15000 )

p_trace = mcmc.trace( "freq_cheating" )[:]
fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(0,1)
ax.set_ylim(auto=True)
ax.hist( p_trace
        , histtype = "stepfilled"
        , normed=True
        , alpha = 0.85
        , bins = 30
        , label='Posterior Distr'
        , color='#348ABD' )
ax.legend(loc='upper right')
ax.vlines( [0.05, 0.35], [0,0], [5,5], alpha = 0.3 )
fig.show()






