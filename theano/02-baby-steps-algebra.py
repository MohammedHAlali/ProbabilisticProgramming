import numpy
import theano.tensor as T
from theano import function
from theano import pp

x = T.dscalar('x')
y = T.dscalar('y')

z = x + y

f = function( [x,y], z )

f(2, 3)

numpy.allclose( f(16.3, 12.1) , 28.4)

type(x)

x.type

T.dscalar

pp(z)

z.eval( { x:16.3, y:12.1} )

# matricies

x=T.dmatrix('x')
y=T.dmatrix('y')
z = x+y
f = function( [x,y], z )

f([[1,2],
   [3,4]], 
  [[10,20],
   [30,40]])

f(numpy.array([[1,2], [3,4]]), 
  numpy.array([[10,20], [30,40]]) )

a = T.vector()
out = a + a**10
f = function( [a], out )
f([0,1,2])

b = T.vector()
out = a**2 + b**2 + 2*a*b
f = function( [a,b], out)

f([0,1,2], [-1,-2,3])

f([0,1,2], [0,0,0])
