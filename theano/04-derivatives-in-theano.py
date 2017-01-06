import theano 
import theano.tensor as T
from theano import pp
import numpy as np

x=T.dscalar( 'x' )
y=x**2
gy = T.grad( y, x )

pp(gy)

f = theano.function( [x], gy )

f( 4 )

f( 94.2 )

pp( f.maker.fgraph.outputs[0] )

# derivative of logistic
x = T.dmatrix('x')
s = T.sum( 1 / (1+T.exp(-x)))
gs = T.grad( s, x )
dlogistic = theano.function( [x], gs)

dlogistic( [[0,1],[-1,-2]] )

# Jacobian
x = T.dvector('x')
y = x**2

J, updates = theano.scan(
        lambda i,y,x : T.grad(y[i], x)
        , sequences=T.arange(y.shape[0])
        , non_sequences=[y,x]
        )

f = theano.function([x], J, updates=updates)
f([4, 4])


# R-operator
W,V = T.dmatrices( 'W', 'V' )
x = T.dvector( 'x' )
y = T.dot( x, W )
JV = T.Rop( y, W, V )
f = theano.function( [W,V,x], JV )
f([[1, 1], [1, 1]], [[2, 2], [2, 2]], [0,1])
