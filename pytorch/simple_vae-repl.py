# imports {{{
%matplotlib
import scipy
import torch
import torch.nn.functional as F
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
# VAE State {{{
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

def plot_elbo_density():
    X_grid = torch.FloatTensor(np.dstack((xx,yy)).reshape(-1,2))
    X_grid = Variable(X_grid)
    vae.num_samples=100
    elbo = vae.ELBO(X_grid).data.numpy()
    elbo = elbo.reshape(xx.shape)
    plt.clf()
    plt.pcolormesh(xx,yy,np.exp(elbo), cmap='viridis')
    plt.scatter(X[0],X[1], color='r', marker='.')
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

# VAE Parameter {{{

model = simple_vae.BarberExample2(10000,2)
def plot_samples():
    try:
        z_samples = model.z.data.numpy().T
    except AttributeError:
        z_samples = model.sample()
    plt.clf()
    plot_generative()
    plt.scatter(z_samples[0], z_samples[1], alpha=0.6)
    plt.xlabel('mu')
    plt.ylabel('logvar')
    plt.title('qz')
    # plt.xlim(-8,8)
    # plt.ylim(-4,4)
    mu, logvar =  torch.unbind(model.qz.mu, 1)
    mu = mu.data.numpy()
    logvar = logvar.data.numpy()
    plt.scatter(mu, logvar, color='black', alpha=0.3)
    plt.pause(0.01)
plot_samples()

plt.clf()
z_samples = model.sample()
sns.kdeplot(z_samples[0], z_samples[1], kind="kde", shade = True)

aux_samples = model.aux_samples.data.numpy().T
plt.clf()
plt.scatter(aux_samples[0], aux_samples[1])

plt.clf()
plt.hist(aux_samples[0])
plt.hist(aux_samples[1])


plt.close()
sns.jointplot(z_samples[0], z_samples[1], kind='hex')

model = simple_vae.BarberExample2(1000,4)
model.annealing = 0
hist = []
hist_lps_list = []
optimizer = torch.optim.Adam(model.parameters()
        , betas=(0.9,0.99)
        , lr=0.001
        , weight_decay=0.001)
plot_samples()


for epoch in range(500):
    model.train(True)
    optimizer.zero_grad()
    elbo = model.ELBO()
    if np.isnan(elbo.data[0]):
        break
    loss = -elbo
    loss.backward()
    optimizer.step()
    model.train(False)
    hist.append(loss.data[0])
    print(epoch, loss.data[0])
    lps = np.array([
        model.px_logpdf(model.z).sum().data[0],
        model.pa.logpdf(model.aux_samples).sum().data[0],
        -model.qa.logpdf(model.aux_samples).sum().data[0],
        -model.qz.logpdf(model.z).sum().data[0] 
    ]) / model.num_samples
    hist_lps_list.append(lps)
    # plot_samples()
hist_lps = np.array(hist_lps_list).T
plt.clf()
plt.plot(hist[:])

plt.clf()
plt.
plt.hist(model.pa.mu.data.numpy().squeeze())
plt.hist(model.pa.logvar.data.numpy().squeeze())

plt.clf()
plt.plot(hist_lps[0], lw=5, alpha=0.7, label='p(x|z)')
plt.plot(hist_lps[1], lw=5, alpha=0.7, label='p(a|z)')
plt.plot(hist_lps[2], lw=5, alpha=0.7, label='q(a)=N(0,I)')
plt.plot(hist_lps[3], lw=5, alpha=0.7, label='$q(z|a)$')
plt.legend()
# plt.ylim(-20,5)



plt.clf()
plt.hist(model.qz.logvar.data.numpy())

model.qz.mu_param

model.qz.logvar_param

model.pa.mu_param.data.numpy()

model.pa.logvar_param.data.numpy()


def generative(mu, logvar):
    # logvar = logvar * np.log(10)
    z = torch.FloatTensor([[mu, logvar]])
    lp = model.generative(z)
    return lp[0]
generative(0,0)

xx, yy = np.meshgrid(
        np.linspace(-8, 4, 400),
        np.linspace(-8, 4, 400))
lp = [[generative(mu, logvar) 
        for mu in xx[1]]
        for logvar in yy[:,1]]
lp = np.array(lp)
nlp = lp - np.max(lp)
pdf = np.exp(lp)
pdf = pdf / np.max(pdf)
lpr = nlp.reshape(-1)
plot_generative()

plt.clf()
plt.hist(lpr, bins=200)
# plt.hist(lpr[lpr>-20], bins=200)

def plot_generative():
    plt.clf()
    # plt.contour(xx,yy,lp,np.linspace(-20,-6,20),cmap='viridis')
    plt.contour(xx,yy,nlp*10**-model.annealing,np.linspace(-10,0,50),cmap='viridis')
    plt.xlabel('mu')
    plt.ylabel('logvar')
plot_generative()

plt.clf()
plt.hist(np.exp(lpr[lpr>-20]))

plt.clf()
plt.imshow(np.exp(lp),cmap='viridis')
plt.clim(0,0.0001)

def lp_gaussian(x, mu, var):
    x = torch.FloatTensor([x])
    mu = torch.FloatTensor([mu])
    var = torch.FloatTensor([var])
    lp = model.logpdf_Gaussian( x, mu, var)
    return lp[0]

lp_gaussian(0,0,10)


def plot_logpdf_Gaussian():
    y2 = -6
    logvar = np.arange(-4,4,0.05)
    logprior = np.array([lp_gaussian(lv, 0, 1) for lv in logvar])
    def plot_logvar(mu):
        lp = np.array([lp_gaussian(y2, mu, np.exp(lv)) for lv in logvar])
        plt.plot(logvar, lp+logprior, label='mu = {}'.format(mu))
        plt.ylim(-20,3)
    plt.clf()
    plt.subplot(2,1,1)
    plt.plot(logvar, logprior, label='prior', lw=5)
    plot_logvar(0)
    plot_logvar(-6)
    plt.xlabel('logvar')
    plt.legend()
    mu = np.arange(-8,8,0.05)
    logprior = np.array([lp_gaussian(m,0,1) for m in mu])
    def plot_mu(var):
        lp = np.array([lp_gaussian(y2, m, var) for m in mu])
        lp = lp + logprior
        plt.plot(mu, lp, label='var = {}'.format(var))
        plt.ylim(-30,0)
    plt.subplot(2,1,2)
    plt.plot(mu, logprior, label='prior', lw=5)
    plot_mu(0.1)
    plot_mu(1.0)
    plot_mu(10)
    plt.xlabel('mu')
    plt.legend()
plot_logpdf_Gaussian()

xx, yy = np.meshgrid(np.arange(-8.0,8,0.05),np.arange(-4,4,0.05))
X_grid = torch.FloatTensor(np.dstack((xx,np.power(10,yy))).reshape(-1,2))
X_grid = Variable(X_grid)

%matplotlib
def pdf_qar():
    pdf = qar.logpdf(X_grid).data.numpy()
    pdf = pdf.reshape(xx.shape)
    plt.clf()
    plt.pcolormesh(xx,yy,np.exp(pdf), cmap='viridis')
pdf_qar()

s = np.array([qar.sample().data[0].numpy() for _ in range(100)]).T
plt.clf()
plt.scatter(s[0], s[1])

r_rng = np.linspace(-4,2,200)
ig = [logpdf_IG(torch.FloatTensor([np.power(10,r)]),v,beta).numpy() for r in r_rng]
ig = np.array(ig)
rg = [logpdf_Gaussian(torch.FloatTensor([r]),-2.3,torch.FloatTensor([.15])).numpy() for r in r_rng]
plt.clf()
plt.scatter(r_rng, np.exp(ig))
plt.scatter(r_rng, 300*np.exp(rg))


a_rng = np.linspace(-8,8)
mu_a = [logpdf_Gaussian(Variable(torch.FloatTensor([a])),0,q).data.numpy() for a in a_rng]
mu_a = np.array(mu_a)
plt.clf()
plt.scatter(a_rng, mu_a)

def log_prior(a, logr):
    a = Variable(torch.FloatTensor([a]))
    logr = Variable(torch.FloatTensor([logr]))
    logr_var = Variable(torch.FloatTensor([0.1]))
    lp = logpdf_Gaussian(a,0,q) + logpdf_Gaussian(logr,0,logr_var)
    return lp.data[0]

mesh = np.meshgrid(a_rng, np.exp(r_rng))
lp = np.zeros_like(mesh[0])
for i in range(a_rng.shape[0]):
    for j in range(r_rng.shape[0]):
        lp[j,i] = log_prior(a_rng[i], r_rng[j])

plt.clf()
plt.contour(mesh[0],mesh[1],lp,cmap='viridis')


ig = simple_vae.InverseGamma()

r_rng = np.logspace(-4,4,100)
lp_ig = [ig.logpdf(r) for r in r_rng]
plt.clf()
plt.xscale('log')
plt.plot(r_rng, lp_ig)


def generative(self, mu, logvar):
    y2 = -6
    var = logvar.exp()
    q = 1
    lp = (self.logpdf_Gaussian(y2, mu, var)
          + self.logpdf_Gaussian(mu, 0, q)
          + ig.logpdf(var))
    return lp

num_samples = 100
aux_dim = 20
hidden_dim = 30
output_dim = 2
fc1 = torch.nn.Linear(aux_dim, hidden_dim)
s = Variable(torch.randn((num_samples, aux_dim)))
h = fc1(s)
hr = torch.nn.functional.relu( h )
fc2 = torch.nn.Linear(hidden_dim, output_dim, bias=False)
fc3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)
out = fc2(hr)

print(fc1.weight)
print(fc1.bias)
print(fc2.weight)
print(fc2.bias)

print(h)
print(hr)

out = fc2(hr)
x,y = out.data.numpy().T
sns.jointplot(x,y)

plt.clf()
plt.matshow(hr.data.numpy())

plt.clf()
plt.title('hidden layer')
plt.subplot(2,1,1)
plt.plot(h[:,0].data.numpy(), marker='o', lw=0)
plt.plot(hr[:,0].data.numpy(), marker='.', lw=0)
plt.xlabel('samples')
plt.subplot(2,1,2)
plt.plot(h[0,:].data.numpy(), marker='o', lw=0)
plt.plot(hr[0,:].data.numpy(), marker='.', lw=0)
plt.xlabel('hidden_dim')


plt.clf()
out = fc2(hr)
x,y = out.data.numpy().T
plt.scatter(x,y,marker='o', s=1000, facecolors='none', edgecolors='r')
out = fc2(h)

x,y = out.data.numpy().T
plt.scatter(x,y,marker='o')
out = []
for i in range(hidden_dim):
    idx = np.zeros(hidden_dim)
    idx[i] = 1
    fc3.weight.data = fc2.weight.data * torch.FloatTensor([idx,idx])
    o = fc3(hr)
    out.append(o)
    x,y = o.data.numpy().T
    plt.plot(x,y,marker='.')
# x,y = (out[0] + out[4]).data.numpy().T
# plt.scatter(x,y,marker='o', s=1000, facecolors='none', edgecolors='r')

dmlp = simple_vae.DeepMLP(100)
out = dmlp.sample()
layer_var = []
for idx, h in enumerate(dmlp.h):
    var = h.data.numpy().var(0)
    for v in var:
        layer_var.append([idx, v])
layer_var = np.array(layer_var)
plt.clf()
plt.scatter(*layer_var.T)
plt.plot((0,len(dmlp.h)),(1,1), color='red')

plt.clf()
for i in range(len(dmlp.h)):
    plt.subplot(len(dmlp.h),1,i+1)
    plt.hist(dmlp.h[i].data.numpy(), bins=60)
    plt.ylabel(i)
    plt.xlim(-4,4)
plt.tight_layout()



N = 1000
d = []
for _ in range(1000):
    y = np.random.randn(N)
    x = F.elu(Variable(torch.Tensor(y))).data.numpy()
    w = np.random.randn(N) * np.sqrt(2/(N*(1+.00)))
    d.append(np.dot(x,w))
np.array(d).var()



dmlp.fc[3].bias



ndist = simple_vae.ReparamNormal_MuLogvar()

ndist(torch.zeros(10,2))

ndist.condition(torch.zeros(10))

ndist.sample()

x = torch.zeros(3,10)

torch.expand(ndist.mu_param,3)

ndist.mu_param.expand(10)

torch.ones(10).expand(10,2)

def fucn(**kwargs):
    mu = kwargs.get('mu', torch.zeros())

torch.zeros(10).unsqueeze(0)

x = torch.ones(4,3)

x.mul_(torch.zeros(4,3)).add_(torch.ones(4,3))

plt.clf()
input_dim = 1
hidden_dim = 10
num_samples = 10000
x = Variable(torch.randn(num_samples,input_dim))# {{{
plt.subplot(411)
plt.title('x')
plt.hist(x.data.numpy(), bins=30)# }}}
fc1 = torch.nn.Linear(input_dim, hidden_dim)# {{{
fc1.bias.data = 0.5*torch.ones(hidden_dim) + 20.0*torch.randn(hidden_dim)
fc1.weight.data = 10*torch.ones(hidden_dim,input_dim)
h = fc1(x)
plt.subplot(412)
plt.title('fc(x)')
plt.hist(h.data.numpy(), bins=30, stacked=True)# }}}
h = F.sigmoid(h)# {{{
plt.subplot(413)
plt.title('tanh(h)')
plt.hist(h.data.numpy(), bins=30)# }}}
h = torch.sum(h,1)# {{{
plt.subplot(414)
plt.title('sum(h)')
plt.hist(h.data.numpy(), bins=300)# }}}

plt.clf()
input_dim = 2
hidden_dim = 10
num_samples = 10000
x = Variable(torch.randn(num_samples,input_dim))# {{{
plt.subplot(411)
plt.title('x')
plt.scatter(*x.data.numpy().T,alpha=0.1)#}}}
fc1 = torch.nn.Linear(input_dim, hidden_dim)# {{{
fc1.bias.data = 0.5*torch.ones(hidden_dim) + 20.0*torch.randn(hidden_dim)
fc1.weight.data *= 70 #*torch.ones(hidden_dim,input_dim)
h = fc1(x)
plt.subplot(412)
plt.title('fc(x)')
plt.hist(h.data.numpy())# }}}
h = F.sigmoid(h)# {{{
plt.subplot(413)
plt.title('sigmoid(h)')
plt.hist(h.data.numpy(), bins=30)# }}}
fc2 = torch.nn.Linear(hidden_dim,input_dim)# {{{
h = fc2(h)
plt.subplot(414)
plt.title('fc2(h)')
sns.kdeplot(*h.data.numpy().T, kind="hex", shade = True)
# plt.scatter(*h.data.numpy().T)# }}}


# # }}}
