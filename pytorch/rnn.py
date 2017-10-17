# https://github.com/pytorch/tutorials/blob/master/Introduction%20to%20PyTorch%20for%20former%20Torchies.ipynb
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter


class RNN(nn.Module):
    # you can also accept arguments in your model constructor
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        input_size = data_size + hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output


rnn = RNN(50, 20, 10)

loss_fn = nn.MSELoss()

batch_size = 10
TIMESTEPS = 5

# Create some fake data
batch = Variable(torch.randn(batch_size, 50))
hidden = Variable(torch.zeros(batch_size, 20))
target = Variable(torch.zeros(batch_size, 10))

loss = 0
for t in range(TIMESTEPS):
    # yes! you can reuse the same network several times,
    # sum up the losses, and call backward!
    hidden, output = rnn(batch, hidden)
    loss += loss_fn(output, target)
loss.backward()


class Model(torch.nn.Module):
    def __init__(self, TIMESTEPS):
        super().__init__()
        self.dt = 0.01
        self.TIMESTEPS = TIMESTEPS
        self.W = torch.nn.Linear(2, 2)
        self.X = Parameter(torch.ones(self.TIMESTEPS, 2))
        self.ic = Variable(torch.FloatTensor([[1, 2]]))

    def dynamics(dt, W, X):
        return X + dt*X*W(X)

    def forward(self):
        F = Model.dynamics(self.dt, self.W, self.X)
        return F


class ModelDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dt = 0.01
        self.W = torch.nn.Linear(2, 2)
        self.ic = Variable(torch.FloatTensor([[1, 2]]))

    @staticmethod
    def dynamics(dt, W, X):
        return X + dt*X*W(X)

    def forward(self, X):
        F = ModelDynamics.dynamics(self.dt, self.W, X)
        return F


#
# simulation
#
dt = 0.01
TIMESTEPS = 2000

target_model = ModelDynamics()
target_model.W.bias.data[0] = 1.0
target_model.W.bias.data[1] = -1.0
target_model.W.weight.data = torch.FloatTensor([[0, -1], [1, 0]])


def simulate(model):
    X = [Variable(torch.zeros(1, 2), requires_grad=False)
         for _ in range(TIMESTEPS)]
    X[0].data = torch.FloatTensor([[1, 2]])
    for t in range(1, TIMESTEPS):
        X[t] = model(X[t-1])
    X = torch.cat(X).detach()
    return X


X_data = simulate(target_model)


print('**************************************************************Reloaded')
