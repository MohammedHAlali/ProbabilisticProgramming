import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.ion()


#
# simulation
#
l1 = torch.nn.Linear(2,2)
l1.weight.data = torch.FloatTensor([[0, -1],[1, 0]])
l1.bias.data[0] = 1.0
l1.bias.data[1] = -1.0

dt = 0.01
TIMESTEPS=2000
X = [Variable(torch.zeros(1,2)) for _ in range(TIMESTEPS)]
X[0].data = torch.FloatTensor([[1,2]])
for t in range(1, TIMESTEPS):
    X[t] = X[t-1] + dt*X[t-1]*l1(X[t-1])
x_data = torch.cat(X).data.numpy()
plt.clf()
plt.scatter(x_data[:,0], x_data[:,1], marker='.')

plt.clf()
plt.subplot(2,1,1)
plt.plot(x_data[:,0])
plt.subplot(2,1,2)
plt.plot(x_data[:,1])


#
# loss function
#
loss_fn = torch.nn.MSELoss()

loss_fn.forward

getattr(loss_fn._backend, type(loss_fn).__name__)

torch.nn._functions.thnn.auto.MSELoss

#
# Learning
#
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() 
        self.l1 = torch.nn.Linear(2,2)
        self.l1.bias.data *= 1e-3
        self.l1.weight.data *= 1e-3
        self.ic = Variable(torch.FloatTensor([[1,2]]))
        self.X = [Variable(torch.zeros(1,2)) for _ in range(TIMESTEPS-1)]
        self.X.insert(0, self.ic)
    def forward(self):
        for t in range(1, TIMESTEPS):
            self.X[t] = self.X[t-1] + dt*self.X[t-1]*self.l1(self.X[t-1])
        Xc = torch.cat(self.X).t()
        X0, X1 = torch.unbind(Xc)
        return X0
model = Model()
model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
x_data_v = Variable(torch.FloatTensor(x_data[:,0]))

for p in model.parameters():
    print(p)

vars = model.state_dict()
vars['ic']

for epoch in range(5000):
    optimizer.zero_grad()
    X0 = model()
    loss = loss_fn(X0, x_data_v)
    print(loss.data[0])
    loss.backward()
    optimizer.step() 
    x = torch.cat(model.X).data.numpy()
    plt.clf()
    plt.scatter(x_data[:,0], x_data[:,1], marker='.')
    plt.scatter(x[:,0], x[:,1], marker='.')
    plt.title(epoch)
    plt.pause(0.05)


plt.clf()
plt.subplot(2,1,1)
plt.plot(x_data[:,0])
plt.plot(x[:,0])
plt.subplot(2,1,2)
plt.plot(x_data[:,1])
plt.plot(x[:,1])


#
# 
#
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() 
        self.l1 = torch.nn.Linear(2,2)
        self.l1.bias.data *= 1e-3
        self.l1.weight.data *= 1e-3
        self.ic = Variable(torch.FloatTensor([[1,2]]))
    def forward(self, x0, x1):
        x0 = torch.FloatTensor([float(x0)])
        x0 = Variable(x0)
        X = torch.cat([x0,x1]).unsqueeze(0)
        X = X + dt*X*self.l1(X)
        x0, x1 = torch.unbind(X.squeeze())
        return x0, x1
model = Model()
x1 = Variable(torch.FloatTensor([2]))
model(x_data[0,0], x1)

model.l1.weight

model.l1.bias


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    loss = 0.0
    x1 = Variable(torch.FloatTensor([2]))
    for t in range(TIMESTEPS-1):
        x0, x1 = model(x_data[t,0], x1)
        x0_1 = Variable(torch.FloatTensor([float(x_data[t+1,0])]))
        loss += loss_fn(x0, x0_1) 
    print(loss.data[0])
    loss.backward()
    optimizer.step() 
    hist = []
    x0 = Variable(torch.FloatTensor([float(x_data[0,0])]))
    x1 = Variable(torch.FloatTensor([2]))
    for t in range(TIMESTEPS):
        x0, x1 = model(x0.data[0],x1)
        hist.append((x0.data[0], x1.data[0]))
    hist = np.array(hist)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(x_data[:,0])
    plt.plot(hist[:,0])
    plt.ylim(-1,5)
    plt.subplot(2,1,2)
    plt.plot(x_data[:,1])
    plt.plot(hist[:,1])
    plt.title(epoch)
    plt.pause(0.05)

    plt.clf()
    plt.scatter(x_data[0:-2,0], x_data[1:-1,0])

    plt.clf()
    plt.scatter(x_data[:,0], x_data[:,1], marker='.')
    plt.scatter(hist[:,0], hist[:,1], marker='.')
    plt.title(epoch)
    plt.pause(0.05)



#
# collocation
#
%load_ext autoreload
%autoreload 2
# %pdb on

x_data_v = Variable(torch.FloatTensor(x_data[:,0]))

import rnn

model = rnn.Model(TIMESTEPS)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    F = model()
    lossD = torch.pow(model.X[1:,:] - F[:-1,:], 2)
    lossD = torch.sum(lossD)
    lossF = torch.pow(model.X[:,0] - x_data_v, 2)
    lossF = torch.sum(lossF)
    # lossD = loss_fn(model.X[1:,:], F[:-1,:])
    # lossF = 100*loss_fn(model.X[:,0], x_data_v)
    loss = lossD + lossF
    print(epoch, lossD.data[0], lossF.data[0], loss.data[0])
    loss.backward()
    optimizer.step() 
    x = torch.unsqueeze(model.X[0],0)
    hist = [x]
    for t in range(TIMESTEPS):
        x = rnn.Model.dynamics( dt, model.l1, x)
        hist.append(x)
    hist = torch.cat(hist)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(x_data[:,0])
    plt.plot(model.X[:,0].data.numpy())
    plt.plot(hist[:,0].data.numpy())
    plt.ylim(-1,5)
    plt.subplot(2,1,2)
    plt.plot(x_data[:,1])
    plt.plot(model.X[:,1].data.numpy())
    plt.plot(hist[:,1].data.numpy())
    plt.title(epoch)
    plt.pause(0.01)


    model.l1.weight.data.numpy().flatten()
    model.l1.bias.data.numpy().flatten()

model.X[0,1].data[0] = 2.5
