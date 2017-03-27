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
        self.l1 = torch.nn.Linear(2, 2)
        self.l1.bias.data *= 1e-3
        self.l1.weight.data *= 1e-3
        self.X = Parameter(torch.ones(self.TIMESTEPS, 2))
        self.ic = Variable(torch.FloatTensor([[1, 2]]))

    def dynamics(dt, l1, X):
        return X + dt*X*l1(X)

    def forward(self):
        F = Model.dynamics(self.dt, self.l1, self.X)
        return F
