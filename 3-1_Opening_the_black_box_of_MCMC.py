import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ch 3: Opening the black box of MCMC

# The Bayesian landscape
# N-unknowns: prior probability surface

# Uniform priors
# p_1 and p_2 ~ Uniform( 0, 5 )
jet = plt.cm.jet
x = y = np.linspace( 0, 5, 100 )
X, Y = np.meshgrid( x, y )
uni_x = stats.uniform.pdf( x, loc=0, scale=5 )
uni_y = stats.uniform.pdf( y, loc=0, scale=5 )
M = np.dot( uni_x[:,None], uni_y[None,:] )

# fig: joint( x1, x2 ) = p_1(x1) * p_2(x2) {{{
fig = plt.figure(1)
fig.clear()
fig.add_subplot( 121 )
ax = fig.gca()
ax.imshow( M
        , interpolation='none'
        , origin='lower'
        , cmap = jet 
        , vmax = 1
        , vmin = 0.15
        , extent= (0, 5, 0, 5)
        )
ax.set_title("Landscape formed by Uniform priors")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
fig.add_subplot( 122 , projection = '3d')
ax = fig.gca()
ax.plot_surface( X, Y, M
        , cmap = plt.cm.jet
        , vmax = 1
        , vmin = 0.15
        )
ax.view_init( azim=390 )
fig.canvas.draw()
fig.show()
# }}}

# Exponential priors
# p_1 ~ Exp( 3 )
# p_2 ~ Exp( 10 )
exp_x = stats.expon.pdf( x, scale = 3 )
exp_y = stats.expon.pdf( x, scale = 10 )
M = np.dot( exp_x[:,None], exp_y[None,:] )

# fig: joint( x1, x2 ) = p_1(x1) * p_2(x2) {{{
fig = plt.figure(1)
fig.clear()
fig.add_subplot( 121 )
ax = fig.gca()
ax.contour( X, Y, M )
ax.imshow( M
        , interpolation='none'
        , origin='lower'
        , cmap = jet 
        , extent= (0, 5, 0, 5)
        )
ax.set_title("Landscape formed by Exponential priors")
ax.set_xlabel("$p_1$")
ax.set_ylabel("$p_2$")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
fig.add_subplot( 122 , projection = '3d')
ax = fig.gca()
ax.plot_surface( X, Y, M
        , cmap = plt.cm.jet
        )
ax.view_init( azim=390 )
fig.canvas.draw()
fig.show()
# }}}

#
# Effect of observed data

# Generate simulated data {{{
# true parameters that we dont know
lambda_1_true = 1
lambda_2_true = 3

# sample size
N = 100

# generate data
data = np.concatenate( [
        stats.poisson.rvs( lambda_1_true, size=(N,1) ), 
        stats.poisson.rvs( lambda_2_true, size=(N,1) )
        ], axis = 1 )
print "sample size: %d" % N, data
# }}}

# likelihood = P[ data | param ]
# L( l1, l2 ) = prod( Poisson( xi; l1 ) * Poisson( xi, l2 ) )
x = y = np.linspace( 0.01, 5, 100 )
likelihood_x = np.array( [
    stats.poisson.pmf(data[:,0], _x) for _x in x] ).prod( axis = 1)
likelihood_y = np.array( [
    stats.poisson.pmf(data[:,1], _y) for _y in y] ).prod( axis = 1)
L = np.dot( likelihood_x[:,None], likelihood_y[None,:] )

# fig: 1D Likelihood {{{
fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("1D Likelihood")
ax.set_xlabel("$\lambda$")
ax.set_ylabel("prob")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.plot( x, likelihood_x, label='likelihood $\lambda_1$'  , color='red' )
ax.plot( x, likelihood_y, label='likelihood $\lambda_2$'  , color='blue' )
ax.legend(loc='upper right')
fig.canvas.draw()
fig.show()
# }}}


# priors are for different parameters lambda of two Poisson distributions 

# prior and posterior plots {{{
uni_x = stats.uniform.pdf( x, loc = 0, scale = 5 )
uni_y = stats.uniform.pdf( x, loc = 0, scale = 5 )
M = np.dot( uni_x[:,None], uni_y[None,:] )

fig = plt.figure(1)
fig.clear()
# Uniform priors {{{
fig.add_subplot( 2,2,1 )
ax = fig.gca()
ax.set_title("Landscape formed by uniform priors on $p_1, p_2$")
ax.set_xlabel("$\lambda_1$")
ax.set_ylabel("$\lambda_2$")
ax.set_xlim(0,5)
ax.set_ylim(0,5)
ax.imshow( M
        , interpolation='none'
        , origin='lower'
        , cmap = jet 
        , extent= (0, 5, 0, 5)
        , vmax = 1
        , vmin = -0.15
        )
ax.scatter( lambda_2_true, lambda_1_true
        , c = "k"
        , s = 50
        , edgecolor = "none"
        )# }}}
# Posterior with Uniform priors {{{
fig.add_subplot( 2,2,3 )
ax = fig.gca()
ax.set_title("Landscape warped by %d data observations;\n\
Uniform priors on $p_1, p_2$" % N)
ax.set_xlabel("$\lambda_1$")
ax.set_ylabel("$\lambda_2$")
ax.set_xlim(0,5)
ax.set_ylim(0,5)
ax.contour( x, y, M * L )
ax.imshow( M * L
        , interpolation='none'
        , origin='lower'
        , cmap = jet 
        , extent= (0, 5, 0, 5)
        )
ax.scatter( lambda_2_true, lambda_1_true
        , c = "k"
        , s = 50
        , edgecolor = "none"
        )# }}}

exp_x = stats.expon.pdf( x, loc=0, scale=3 )
exp_y = stats.expon.pdf( y, loc=0, scale=10 )
M = np.dot( exp_x[:,None], exp_y[None,:] )

# Exponential priors {{{
fig.add_subplot( 2,2,2 )
ax = fig.gca()
ax.set_title("Landscape formed by Exponential priors on $p_1, p_2$")
ax.set_xlabel("$\lambda_1$")
ax.set_ylabel("$\lambda_2$")
ax.set_xlim(0,5)
ax.set_ylim(0,5)
ax.imshow( M
        , interpolation='none'
        , origin='lower'
        , cmap = jet 
        , extent= (0, 5, 0, 5)
        )
ax.scatter( lambda_2_true, lambda_1_true
        , c = "k"
        , s = 50
        , edgecolor = "none"
        )# }}}
# Posterior with Exponential priors {{{
fig.add_subplot( 2,2,4 )
ax = fig.gca()
ax.set_title("Landscape warped by %d data observations;\n\
Exponential priors on $p_1, p_2$" % N)
ax.set_xlabel("$\lambda_1$")
ax.set_ylabel("$\lambda_2$")
ax.set_xlim(0,5)
ax.set_ylim(0,5)
ax.contour( x, y, M * L )
ax.imshow( M * L
        , interpolation='none'
        , origin='lower'
        , cmap = jet 
        , extent= (0, 5, 0, 5)
        )
ax.scatter( lambda_2_true, lambda_1_true
        , c = "k"
        , s = 50
        , edgecolor = "none"
        )# }}}

fig.canvas.draw()
fig.show()
# }}}

