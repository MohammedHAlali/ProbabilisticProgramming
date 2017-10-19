%matplotlib notebook
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 10.0)
%load_ext autoreload
%autoreload 2
import rnn


X_data = rnn.X_data.data.numpy()

plt.figure('data') # {{{
plt.clf()
plt.subplot(1,2,1)
plt.scatter(X_data[:,0], X_data[:,1], marker='.')
plt.xlabel('X1')
plt.ylabel('X2')
plt.subplot(2,2,2)
plt.ylabel('X1')
plt.plot(X_data[:,0])
plt.subplot(2,2,4)
plt.ylabel('X2')
plt.plot(X_data[:,1])
plt.tight_layout()
# }}}

loss_fn = torch.nn.MSELoss()

model = rnn.ModelDynamics()
print(model.W.bias.data)
print(model.W.weight.data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

X0 = rnn.X_data[:-1]
X1 = rnn.X_data[1:]
print(loss_fn(model(X0), X1).data[0])
print(loss_fn(rnn.target_model(X0), X1).data[0])

X_sim = rnn.simulate(model)

fig = plt.figure('sim') # {{{
fig.clear()
ax = fig.gca()
ax.plot(X_sim[:,0].data.numpy(), label='X1')
ax.plot(X_sim[:,1].data.numpy(), label='X2')
ax.set_ylim(0, 3)
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()
# }}}



def plot_state(ax, idx):
    ax.clear()
    ax.plot(X_sim[:, idx], label='sim')
    ax.plot(X_data[:, idx], label='data')
    ax.set_ylim(0, 3)


loss_history = []
fig, axs = plt.subplots(3, 1, num='state-loss')

for epoch in range(2000):
    optimizer.zero_grad()
    loss = loss_fn(model(X0), X1)
    loss.backward()
    optimizer.step() 
    loss_history.append(loss.data[0])
    X_sim = rnn.simulate(model).data.numpy()
    plot_state(axs[0], 0)
    plot_state(axs[1], 1)
    axs[2].clear()
    axs[2].semilogy(loss_history, marker='.', linewidth=0)
    fig.canvas.draw()

#
# hidden x_0 state
#
V0 = rnn.X_data[:-1,1].unsqueeze(1)
V1 = rnn.X_data[1:,1]

H = Parameter(0.01*torch.randn(rnn.TIMESTEPS)+1.)
h0 = H[:-1].unsqueeze(1)
h1 = H[1:]
model = rnn.ModelDynamics()
model.W.bias.data = rnn.target_model.W.bias.data
model.W.weight.data = rnn.target_model.W.weight.data
print(model.W.bias.data)
print(model.W.weight.data)
optimizer = torch.optim.SGD([
    {'params': model.parameters(), 'lr': 0.},
    {'params': H, 'lr': 1.01}])
loss_history = []
loss_visible_history = []
loss_hidden_history = []

fig, axs = plt.subplots(4, 1, num='state-loss')

fig.clear()

V0[200]

    idx = 200
    h0.data[idx] = 2.0
    X0 = torch.cat((h0, V0), 1)
    X1_model = model(X0)
    print('X1: ', X1_model.data[idx,0], X1_model.data[idx,1])
    print('h1: ', h1.data[idx])
    print('V1: ', V1.data[idx])
    d_hidden = (X1_model[:,0] - h1)**2
    d_visible = (X1_model[:,1] - V1)**2
    print('dH: ', d_hidden.data[idx])
    print('dV: ', d_visible.data[idx])

H.grad[1:10] * optimizer.param_groups[1]['lr']

for epoch in range(4000):
    optimizer.zero_grad()
    H.data[0] = 1.0
    X0 = torch.cat((h0, V0), 1)
    X1_model = model(X0)
    d_hidden = (X1_model[:,0] - h1)**2
    d_visible = (X1_model[:,1] - V1)**2
    loss_hidden = torch.sum(d_hidden) * 0 / rnn.TIMESTEPS
    loss_visible = torch.sum(d_visible) * 1 / rnn.TIMESTEPS
    loss = loss_visible + loss_hidden
    loss.backward()
    H.grad.data[0] = 0.0
    # __import__('IPython.core.debugger').core.debugger.set_trace()
    optimizer.step()
    loss_history.append(loss.data[0])
    loss_visible_history.append(loss_visible.data[0])
    loss_hidden_history.append(loss_hidden.data[0])
    if epoch % 10 != 0:
        continue
    X_sim = rnn.simulate(model).data.numpy()
    axs[0].clear()
    axs[0].set_ylim(0, 3)
    axs[0].plot(X_sim[:, 0], label='sim')
    axs[0].plot(rnn.X_data[:, 0].data.numpy(), label='data')
    axs[0].plot(np.arange(1,rnn.TIMESTEPS),X1_model[:,0].data.numpy(), label='model', linewidth=4, linestyle='--')
    axs[0].plot(H.data.numpy(), label='colloc')
    axs[0].legend()
    axs[1].clear()
    axs[1].set_ylim(0, 3)
    axs[1].plot(X_sim[:, 1], label='sim')
    axs[1].plot(rnn.X_data[:, 1].data.numpy(), label='data')
    axs[1].plot(X1_model[:,1].data.numpy(), label='model', linestyle='--')
    axs[1].legend()
    axs[2].clear()
    axs[2].semilogy(loss_visible_history, marker='.', linewidth=0, label='visible')
    axs[2].semilogy(loss_hidden_history, marker='.', linewidth=0, label='hidden')
    axs[2].semilogy(loss_history, marker='o', linewidth=0, label='loss')
    axs[2].legend()
    axs[3].clear()
    axs[3].set_ylim(-0.0001, 0.0003)
    axs[3].plot(d_visible.data.numpy(), label='visible')
    axs[3].plot(d_hidden.data.numpy(), label='hidden')
    axs[3].legend()
    fig.canvas.draw() 



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
    lossD = 1000*torch.sum(lossD)
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




for i in range(10):
    plt.clf()
    plt.plot([4,-i,2])
    plt.gcf().canvas.draw()



#
# linear regression y = x^2 {{{
batch_size = 1
num_samples = 100
input = Variable(torch.randn(batch_size, num_samples))
target = input**2

fig = plt.figure(1)
fig.clear()
ax = fig.gca()
ax.set_title("")
ax.set_xlabel("input")
ax.set_ylabel("target")
ax.set_xlim(auto=True)
ax.set_ylim(auto=True)
ax.scatter( input.squeeze(0).data.numpy(), target.squeeze(0).data.numpy() , label=''  , color='red' )
ax.legend(loc='upper left')
fig.canvas.draw()
fig.show()

W = torch.nn.Linear(10,1)

W(input)

d = torch.nn.Dropout

m = torch.nn.Dropout(p=0.99)
input = Variable(torch.randn(7,7))
output = m(input)
output
# }}}
