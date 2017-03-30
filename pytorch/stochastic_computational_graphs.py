import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.ion()

# Variable is a node in a computational graph
# x = Variable()
# x.data is a Tensor
# x.grad is a Variable



x.backward(torch.FloatTensor([2]), retain_variables=True)

x.backward(retain_variables=True)

x.backward(torch.FloatTensor([10]))
theta.grad

theta.grad.data.zero_()


def stochgraph(input):
    theta = Variable(torch.FloatTensor([input]), requires_grad=True)
    x = (theta-1).pow(2)
    r = Variable(torch.randn(1), requires_grad=False)
    y = x + 0*r
    f = (y - 5/2).pow(2)
    return f


input_rng = np.arange(-2., 4., 0.01)
pts = [stochgraph(input).data[0] for input in input_rng]
plt.clf()
plt.scatter(input_rng, pts, marker='.')

input_rng[142]
# -0.57999

learning_rate = 1e-2
input = -2
theta = Variable(torch.FloatTensor([input]), requires_grad=True)
hist = []
optimizer = torch.optim.Adam([theta], lr=learning_rate)
for ti in range(1000):
    x = (theta-1).pow(2)
    r = Variable(torch.randn(1), requires_grad=False)
    y = x + r
    f = (y - 5/2).pow(2)
    optimizer.zero_grad()
    f.backward()
    # theta.data -= 1./ti * theta.grad.data
    optimizer.step()
    hist.append(theta.data[0])
plt.clf()
plt.plot(hist)
plt.hlines(-0.5799,-1,1e3)


class StochGraph(torch.nn.Module):
    def __init__(self, input):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.FloatTensor([input]))
        self.r = Variable(torch.randn(1), requires_grad=False)

    def forward(self):
        r.data = torch.randn(1)
        x = (self.theta-1).pow(2)
        y = x + r
        f = (y - 5/2).pow(2)
        return f


model = StochGraph(-2)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

list(model.parameters())

for ti in range(1000):
    f_pred = model()
    optimizer.zero_grad()
    f_pred.backward()
    optimizer.step()

model.theta.data




torch.nn.Sequential



x = Variable(torch.ones(1), requires_grad=True)
y = x*x
y.backward(torch.FloatTensor([8]))
x.grad


