import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

%load_ext autoreload
%autoreload 2
%pdb on
%pdb off

import reparameterized_regression


N = 10
input_data = np.random.uniform(4,6,N)
output_data = 3*input_data-2 + np.random.randn(N)
plt.clf()
plt.xlim(0,10)
plt.ylim(-3,30)
plt.scatter(input_data, output_data)

from sklearn import linear_model
linreg = linear_model.LinearRegression()
linreg.fit(input_data.reshape(-1,1), output_data)
linreg.coef_
# Out[10]: array([ 2.9758882])
linreg.intercept_
# Out[11]: -2.0223463314976993



def make_var(X):
    X = torch.FloatTensor(X)
    X = torch.unsqueeze(X,1)
    X = Variable(X)
    return X

X_data = make_var(input_data)
Y_data = make_var(output_data)

def show_plot(epoch):
    Y = model.l1(X_data)
    plt.clf()
    plt.xlim(0,10)
    plt.ylim(-3,30)
    plt.title(epoch)
    plt.scatter(input_data, output_data)
    plt.plot(X_data.squeeze().data.numpy(), Y.squeeze().data.numpy(),
    color='lightgreen')
    plt.pause(0.01)
show_plot(1)


model = reparameterized_regression.Model()
learning_rate=0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    output = model(X_data)
    loss = loss_fn(output, Y_data)
    loss.backward()
    optimizer.step()
    show_plot(epoch)
    print( epoch, model.l1.weight.data[0,0], model.l1.bias.data[0])



b = reparameterized_regression.RepNormal()
b()

bayesmodel = reparameterized_regression.BayesianModel()
bayesmodel(X_data)

def show_plot(epoch):
    plt.clf()
    plt.title(epoch)
    plt.ylim(-20,40)
    for _ in range(100):
        X_test = 10*np.random.random(N)
        X_test = Variable(torch.FloatTensor(X_test))
        X_test = X_test.unsqueeze(1)
        Y = bayesmodel(X_test)
        plt.scatter(X_test.squeeze().data.numpy(), Y.squeeze().data.numpy(),
            color='green', alpha=0.1)
    plt.scatter(input_data, output_data)
    plt.pause(0.01)
show_plot(1)

learning_rate=0.02
optimizer = torch.optim.Adam(bayesmodel.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()
L = 10
for epoch in range(200):
    optimizer.zero_grad()
    for _ in range(L):
        output = bayesmodel(X_data)
        loss = loss_fn(output, Y_data) / L
    loss.backward()
    optimizer.step()
    show_plot(epoch)
    print( epoch
            , loss.data[0]
            , bayesmodel.w_param.mu.data[0]
            , bayesmodel.b_param.mu.data[0])

bayesmodel.b_param.log_variance.data[0] = 1

#
#
#
mu = Variable(torch.FloatTensor([1.0]), requires_grad=True)
loss = torch.pow(mu - 3.0, 2)
loss.backward()
mu.grad

x = Variable(torch.FloatTensor([[1.0]]))
l1 = torch.nn.Linear(1,1)
mu = l1(x)
loss = torch.pow(mu - 3.0, 2)
loss.backward()
print("mu.grad: ", mu.grad)
print("w grad: ", l1.weight.grad)


