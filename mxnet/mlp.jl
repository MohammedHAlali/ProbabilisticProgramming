using MXNet

#-- Option 3: using nn-factory
mlp = @mx.chain mx.Variable(:data) =>
  mx.MLP([128, 64, 10])            =>
  mx.SoftmaxOutput(name=:softmax)

# data provider
batch_size = 100
include("/home/abergman/.julia/v0.5/MXNet/examples/mnist/mlp.jl")
train_provider, eval_provider = get_mnist_providers(batch_size)

model = mx.FeedForward(mlp, context=[ mx.cpu()
                                     ,mx.cpu()
                                     ,mx.cpu()
                                     ,mx.cpu()] )
