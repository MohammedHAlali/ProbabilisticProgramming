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


N = 100
input_data = 10*np.random.random(N)
output_data = 3*input_data-2 + np.random.randn(N)
plt.clf()
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


