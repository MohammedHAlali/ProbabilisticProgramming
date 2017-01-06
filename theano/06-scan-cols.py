import theano 
import theano.tensor as T
from theano import pp
from theano.ifelse import ifelse
import numpy as np
import time

# Computing norms of columns of X


# define tensor variable
X = T.matrix("X")
results, updates = theano.scan(lambda x_i: T.sqrt((x_i ** 2).sum()), sequences=[X])

compute_norm_lines = theano.function(inputs=[X], outputs=results)

# test value
x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)
print(compute_norm_lines(x))

# comparison with numpy
print(np.sqrt((x ** 2).sum(1)))

np.diag( [1,2,3], 1 )

def norm_col(x_i):
    T.sqrt((x_i ** 2).sum()) 

results, updates = theano.scan(norm_col, sequences=[X])

