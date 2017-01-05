import numpy
import theano.tensor as T
from theano import function
from theano import pp
from theano import In

x = T.dmatrix('x')
s = 1/(1+T.exp(-x))

logistic = function( [x], s)

logistic( [[0,1],[-1,-2]] )

# multiple outputs
a,b = T.dmatrices( 'a', 'b' )
diff = a - b
abs_diff = abs(diff)
diff_sq = diff**2
f = function( [a,b], [diff,abs_diff,diff_sq] )

f([[1]], [[3]])

# Default Value
x, y= T.dscalars('x','y')
z=x+y
f=function([x, In(y,value=1)], z)

f( 33 )

f(33, 2)

x,y,w=T.dscalars('x','y','w')
z=(x+y)*w
f=function([x,In(y,value=1),In(w,value=2,name='w_by_name')], z)

f(33)

f(33,2)

f(33,w_by_name=1)
