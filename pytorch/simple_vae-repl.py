# imports {{{
import scipy
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

import simple_vae
# }}}

# 2D fake data {{{
def FakeData( N=100, x_shift=0, y_shift=0):
    x1 = np.pi*np.random.rand(N)
    x2 = np.sin(x1+x_shift*np.pi) + 0.1*np.random.randn(N) + y_shift
    return x1, x2
x1 = FakeData()
x2 = FakeData(100, 1.1, -2)
plt.clf()
plt.scatter(x1[0], x1[1])
plt.scatter(x2[0], x2[1])

X = np.concatenate((x1,x2), axis=1)
X = (X.T - X.mean(1) ) / X.std(1)
X = X.T
X.shape
X_var = Variable(torch.FloatTensor(X.T))
plt.clf()
plt.scatter(X[0],X[1], color='r')

xx, yy = np.meshgrid(np.arange(-3.0,3,0.05),np.arange(-3,3,0.05))

# }}}


# old vae model {{{

model = simple_vae.Model(3)
z = torch.randn(1000,model.latent_dim)
z = Variable(z)
x, x_logvar = model.generate(z)
x = x.data.numpy().T
plt.clf()
plt.scatter(x[0], x[1])


model = simple_vae.Model(2)
x_recon, x_logvar, mu, logvar = model(X_var)
x_recon = x_recon.data.numpy()
plt.clf()
plt.scatter(x_recon[0],x_recon[1])
plt.scatter(X[0],X[1], color='r')
optimizer = torch.optim.Adam(model.parameters(),
        lr=0.01,
        weight_decay=0.1)
mseloss = torch.nn.MSELoss(size_average=False)

def loss_fn( x_recon, mu, logvar):
    KL_qp = simple_vae.KL_Gaussians(mu, logvar)
    return mseloss(x_recon, X_var) - KL_qp

def logpdf(x, mu, logvar):
    logpdf_i = (x - mu).pow(2).mul(-0.5/(logvar.exp()))
    return torch.sum(logpdf_i)

hist = []
for epoch in range(100):
    optimizer.zero_grad()
    x_recon, x_logvar, mu, logvar = model(X_var)
    KL_qp = simple_vae.KL_Gaussians(mu, logvar)
    # mse = mseloss(x_recon, X_var) / 0.001
    lp = logpdf( X_var, x_recon, x_logvar)
    elbo =  lp - KL_qp
    loss = -elbo
    # loss =  mse - KL_qp
    # loss = loss_fn(x_recon, mu, logvar)
    loss.backward()
    optimizer.step()
    hist.append(loss.data[0])
    print( epoch, lp.data[0], KL_qp.data[0], loss.data[0] )
    x_recon = x_recon.data.numpy().T
    plt.clf()
    plt.scatter(x_recon[0],x_recon[1])
    plt.scatter(X[0],X[1], color='r')
    z = torch.randn(1000,model.latent_dim)
    z = Variable(z)
    x, x_logvar = model.generate(z)
    x = x.data.numpy().T
    plt.scatter(x[0], x[1], marker='.')
    plt.pause(0.01)
plt.clf()
plt.plot(hist[:])

plt.clf()
plt.hist(model.fc3.weight.data.numpy().squeeze(), bins=33)

w = model.fc3.weight.data.numpy().squeeze()
w = np.abs(w)
w.sort()
plt.clf()
plt.plot(w,linewidth=0, marker='.')


X_grid = torch.FloatTensor(np.dstack((xx,yy)).reshape(-1,2))
X_grid = Variable(X_grid)
pdf = model.marginal_x(X_grid)
plt.clf()
plt.pcolormesh(xx,yy,pdf.reshape(xx.shape), cmap='viridis')
plt.scatter(X[0],X[1], color='r')


z = torch.randn(10000,model.latent_dim)
z = Variable(z)
x = model.generate(z)
x = x.data.numpy().T
plt.clf()
plt.scatter(x[0], x[1], alpha=0.1)
plt.scatter(X[1],X[1], color='r')

mu, logvar = model.posterior(X_var)
mu = mu.squeeze().data.numpy()
mu.sort()
plt.clf()
# plt.scatter(mu[0], mu[1])
plt.plot(mu)

# }}}

#
# ReparamNormal {{{
#
dat = 10*np.random.randn(100000)
plt.clf()
plt.hist(dat, bins=200, normed=True)
x = np.linspace(-40,40,100)
pdf = scipy.stats.norm.pdf(x, scale=10)
plt.plot(x,pdf, lw=9, alpha=0.5)


px = simple_vae.ReparamNormal(1,100,1)
x = np.linspace(-4,4,1000)
x = Variable(torch.FloatTensor(x).unsqueeze(1))
prior_z = simple_vae.StandardNormal((1000,1))
z = prior_z.sample()
px.condition(z)
lp = px(x)
plt.clf()
plt.scatter(x.data.numpy(), lp.exp().data.numpy())


N = 1000
z = np.random.randn(N)
z = (z - z.mean()) / z.std()
x = np.cos(z) + 0.1*np.random.randn(N)
x = np.concatenate([x, x+2])
x = (x - x.mean()) / x.std()
X_var = Variable(torch.FloatTensor(x).unsqueeze(1))
Z_var = Variable(torch.FloatTensor(z).unsqueeze(1))
plt.clf()
plt.scatter(z,x)

plt.clf()
plt.hist(x,bins=100)



qz = simple_vae.ReparamNormal(1,100,1)

qz.condition(X_var)
qz_sample = qz.sample()
plt.clf()
plt.xlabel('x')
plt.ylabel('z')
plt.scatter(x, qz_sample.data.numpy(), alpha=0.5)
plt.scatter(x, qz.mu.data.numpy())
plt.scatter(x, qz.mu.data.numpy() + 2*qz.std.data.numpy())
plt.scatter(x, qz.mu.data.numpy() - 2*qz.std.data.numpy())
kl = qz.KL_StdNormal()
plt.scatter(x, kl.data.numpy())

hist = []
optimizer = torch.optim.Adam(
        qz.parameters(),
        lr=0.01,
        weight_decay=0.001)

for epoch in range(200):
    optimizer.zero_grad()
    qz.condition(X_var)
    kl = qz.KL_StdNormal()
    kl = torch.sum(kl)
    qz_sample = qz.sample()
    lp = torch.sum((qz_sample - X_var).pow(2))
    loss = kl + lp
    print( epoch, loss.data[0])
    hist.append(loss.data[0])
    loss.backward()
    optimizer.step()
plt.clf()
plt.plot(hist)



px.condition(qz_sample)
lp = px(X_var)
plt.clf()
plt.scatter(z, lp.exp().data.numpy(), label='log pdf')

def plot_prediction():
    plt.clf()
    # z_sample = prior_z.sample()
    z_sample = np.linspace(-3,3,800)
    z_sample = Variable(torch.FloatTensor(z_sample).unsqueeze(1))
    px.condition(z_sample)
    z_sample = z_sample.data.numpy()
    x_sample = px.sample()
    plt.xlim(-4,4)
    plt.ylim(-4,4.)
    # plt.scatter(z,x, label='data', marker="s", color='black', alpha=0.1)
    plt.scatter(z_sample, x_sample.data.numpy(), label='sample', alpha=0.3)
    plt.scatter(z_sample, px.mu.data.numpy(), label='mu', marker='.')
    plt.scatter(z_sample, px.std.data.numpy(), label='std')
    plt.legend()
    plt.pause(0.001)
plot_prediction()

model = torch.nn.Module()
model.add_module('px', px)
model.add_module('qz', qz)

hist = []
optimizer = torch.optim.Adam(model.parameters(),
        lr=0.001,
        weight_decay=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    qz.condition(X_var)
    qz_sample = qz.sample()
    px.condition(qz_sample)
    lp = px(X_var)
    kl = qz.KL_StdNormal()
    loss = kl - lp
    loss = torch.sum(loss)
    loss.backward()
    optimizer.step()
    hist.append(loss.data[0])
    print( epoch, loss.data[0] )
    plot_prediction()
plt.clf()
plt.plot(hist[:])

plt.clf()
w = px.fc_logvar.weight.data.numpy().squeeze()
w.sort()
plt.plot(w)

plt.clf()
plt.hist(px.sample().data.numpy(),normed=True,alpha=0.5,bins=30)
plt.hist(x,normed=True,alpha=0.5,bins=30)


# }}}

#
# VAE {{{
#
vae = simple_vae.VAE()
vae(X_var)

std_z = simple_vae.StandardNormal((1000,vae.latent_dim))

def plot_prediction():# {{{
    vae.qz.condition(X_var)
    z = vae.qz.sample()
    vae.px.condition(z)
    x = vae.px.sample().data.numpy().T
    mu = vae.px.mu.data.numpy().T
    z = std_z.sample()
    vae.px.condition(z)
    x_pred = vae.px.sample()
    x_pred = x_pred.data.numpy().T
    plt.clf()
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.scatter(x[0],x[1], alpha=0.5)
    plt.scatter(mu[0],mu[1])
    plt.scatter(X[0],X[1], color='r')
    plt.scatter(x_pred[0],x_pred[1], alpha=0.1)
    plt.pause(0.01)
plot_prediction()
# }}}

hist = []
vae = simple_vae.VAE(1)
vae.num_samples = 10
optimizer = torch.optim.Adam(vae.parameters(),
        lr=0.001,
        weight_decay=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    elbo = vae(X_var)
    loss = -elbo
    loss.backward()
    optimizer.step()
    hist.append(loss.data[0])
    print( epoch, loss.data[0] )
    plot_prediction()
plt.clf()
plt.plot(hist[:])

for epoch in range(20):
    optimizer.zero_grad()
    vae.qz.condition(X_var)
    z = vae.qz.sample()
    vae.px.condition(z)
    lp = vae.px.logpdf(X_var)
    kl = vae.qz.KL_StdNormal()
    loss = torch.sum(kl - lp)
    loss.backward()
    optimizer.step()
    hist.append(loss.data[0])
    print( epoch, loss.data[0] )
    plot_prediction()
plt.clf()
plt.plot(hist[:])

# plot qz_mu, qz_std vs X {{{
vae.qz.condition(X_var)
qz_mu  = vae.qz.mu.data.numpy().squeeze()
qz_std = vae.qz.std.data.numpy().squeeze()
plt.clf()
plt.subplot(2,1,1)
plt.title("qz mu")
plt.xlabel('X')
plt.ylabel('Z mu')
plt.scatter(X[0], qz_mu)
plt.scatter(X[1], qz_mu)
plt.subplot(2,1,2)
plt.title("qz std")
plt.xlabel('X')
plt.ylabel('std')
plt.scatter(X[0], qz_std)
plt.scatter(X[1], qz_std)
# }}}

# plot pz_mu, pz_std vs X {{{
z = vae.qz.sample()
vae.px.condition(z)
px_mu  = vae.px.mu.data.numpy().T
px_std = vae.px.std.data.numpy().T
plt.clf()
plt.subplot(2,1,1)
plt.title("px mu")
plt.xlabel('X0')
plt.ylabel('X1')
plt.scatter(px_mu[0], px_mu[1])
plt.scatter(px_mu[0]+px_std[0], px_mu[1]+px_std[1])
plt.scatter(px_mu[0]-px_std[0], px_mu[1]-px_std[1])
plt.subplot(2,1,2)
plt.title("px std")
plt.xlabel('X0')
plt.ylabel('X1')
plt.scatter(px_std[0], px_std[1])
# }}}

X_grid = torch.FloatTensor(np.dstack((xx,yy)).reshape(-1,2))
X_grid = Variable(X_grid)
vae.num_samples=100
elbo = vae.ELBO(X_grid).data.numpy()
elbo = elbo.reshape(xx.shape)
plt.clf()
plt.pcolormesh(xx,yy,np.exp(elbo), cmap='viridis')
plt.scatter(X[0],X[1], color='r')
x = vae.px.sample()
x = x.data.numpy().T
# plt.scatter(x[0],x[1])
mu = vae.px.mu.data.numpy().T
# plt.scatter(mu[0],mu[1])

plt.clf()
plt.hist(elbo.reshape(-1))


# hist [px qz] x [mu std] {{{
plt.clf()
plt.subplot(2, 2, 1)
plt.title('px std')
plt.hist( vae.px.std.data.numpy() )
plt.subplot(2, 2, 2)
plt.title('qz std')
plt.hist( vae.qz.std.data.numpy() )
plt.subplot(2, 2, 3)
plt.title('px mu')
plt.hist( vae.px.mu.data.numpy() )
plt.subplot(2, 2, 4)
plt.title('qz mu')
plt.hist( vae.qz.mu.data.numpy() )
# }}}

plt.clf()
plt.hist(vae.Expectation(X_var, vae.qz).data.numpy())


px = simple_vae.ReparamNormal()
def std(s):
    px.logvar = torch.FloatTensor([[0,0,0,0]]) + s
    px.var = px.logvar.exp()
    px.std = px.var.sqrt()
    x = torch.randn(4,1)
    return px.logpdf(x).sum()
s_rng = np.linspace(-2,2,20)
lp = [std(s) for s in s_rng]
plt.clf()
plt.scatter(s_rng, lp)


vae.qz.condition(X_var)
z = vae.qz.sample()
vae.px.condition(z)
x = vae.px.sample().data.numpy().T
mu = vae.px.mu.data.numpy().T
plt.clf()
plt.scatter(mu[0],mu[1])
plt.scatter(x[0],x[1])
plt.scatter(X[0],X[1], color='r')

vae.ELBO(X_var)

lp = vae.Expectation(X_var, vae.qz)
torch.sum(lp)

kl = simple_vae.VAE.KL_Gaussians(vae.qz.mu, vae.qz.logvar, vae.qz.var)
torch.sum(kl)

vae.state_dict()

X_var - vae.px.mu

vae.px.logpdf(X_var).sum()

tau = np.linspace(0,2,1000)
shape = 2
scale = 1.0
gamma = np.power(tau,-shape-1) * np.exp(-scale/tau)
plt.clf()
plt.plot(tau, gamma)

# }}}
