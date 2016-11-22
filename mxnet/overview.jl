using MXNet

x = mx.zeros( 2, 3)
y = zeros( eltype(x), size(x) )
copy!( y, x)

zeros(x)

mx.ones( size(x), mx.gpu() )

mx.empty( 10, 20 )

x[:,2]

mx.view(x,:,2)

mx.slice(x,1:2)

a = mx.empty(2,3)

a[:] = 0.5

a[:] = rand(size(a))

copy!( a, rand(size(a)) )

b = mx.empty(size(a))

b = a

summary(copy(b))

a = mx.ones(2)
b = mx.ones(2)

@mx.inplace a += b

copy(a)

#
#
using BenchmarkTools
using MXNet

N_REP = 1000
SHAPE = (128, 64)
CTX   = mx.cpu()
LR    = 0.1

function inplace_op()
  weight = mx.zeros(SHAPE, CTX)
  grad   = mx.ones(SHAPE, CTX)

  # pre-allocate temp objects
  grad_lr = mx.empty(SHAPE, CTX)

  for i = 1:N_REP
    copy!(grad_lr, grad)
    @mx.inplace grad_lr .*= LR
    @mx.inplace weight -= grad_lr
  end
  return weight
end

function normal_op()
  weight = mx.zeros(SHAPE, CTX)
  grad   = mx.ones(SHAPE, CTX)

  for i = 1:N_REP
    weight[:] -= LR * grad
  end
  return weight
end

# make sure the results are the same
@assert(maximum(abs(copy(normal_op() - inplace_op()))) < 1e-6)

@benchmark inplace_op()

@benchmark normal_op()

#
# KV store
kv    = mx.KVStore(:local)
shape = (2,3)
key   = 3

mx.init!(kv, key, mx.ones(shape)*2)
a = mx.empty(shape)
mx.pull!(kv, key, a) # pull value into a
println(copy(a))


#
# Intermediate level interface
A = mx.Variable(:A)
B = mx.Variable(:B)
C = A + B

net = mx.Variable(:data)
net = mx.FullyConnected(data=net, name=:fc1, num_hidden=128)
net = mx.Activation(data=net, name=:relu1, act_type=:relu)
net = mx.FullyConnected(data=net, name=:fc2, num_hidden=64)
net = mx.SoftmaxOutput(data=net, name=:out)

mx.list_arguments( net )

net = mx.Variable( :data )

w = mx.Variable( :myweight )

net = mx.FullyConnected( data=net, weight=w, name=:fc1, num_hidden=128)

mx.list_arguments( net )

net  = mx.Variable(:data)
net  = mx.FullyConnected(data=net, name=:fc1, num_hidden=128)
net2 = mx.Variable(:data2)
net2 = mx.FullyConnected(data=net2, name=:net2, num_hidden=128)
mx.list_arguments(net2)
composed_net = net2(data2=net, name=:composed)
mx.list_arguments(composed_net)

#
# Shape Inference
A = mx.Variable( :A )
B = mx.Variable( :B )
C = A .* B
a = mx.ones(3) * 4
b = mx.ones(3) * 2
c_exec = mx.bind( C, context=mx.cpu(), args=Dict( :A => a
                                                 ,:B => b ) )
mx.forward( c_exec )
copy( c_exec.outputs[1] )
