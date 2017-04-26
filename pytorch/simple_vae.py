import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import scipy
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        h = 100
        self.latent_dim = latent_dim
        self.fc1 = torch.nn.Linear(2, h)
        self.fc21 = torch.nn.Linear(h, latent_dim)
        self.fc22 = torch.nn.Linear(h, latent_dim)
        self.fc3 = torch.nn.Linear(latent_dim, h)
        self.fc41 = torch.nn.Linear(h, 2)
        self.fc42 = torch.nn.Linear(h, 2)
        Model.init_lin(self.fc1)
        Model.init_lin(self.fc21)
        Model.init_lin(self.fc22)
        Model.init_lin(self.fc3)
        Model.init_lin(self.fc41)
        Model.init_lin(self.fc42)

    def init_lin(fc):
        fc.weight.data.normal_(0.0, 0.01)
        fc.bias.data.fill_(0.0)

    def forward(self, x):
        z_mu, z_logvar = self.posterior(x)
        z = self.reparametrize(z_mu, z_logvar)
        x_mu, x_logvar = self.generate(z)
        return x_mu, x_logvar, z_mu, z_logvar

    def posterior(self, x):
        qh = F.relu(self.fc1(x))
        mu = self.fc21(qh)
        logvar = self.fc22(qh)
        return mu, logvar

    def generate(self, z):
        h = F.relu(self.fc3(z))
        mu = self.fc41(h)
        logvar = self.fc42(h)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.exp()
        eps = Variable(torch.randn(mu.size()))
        return mu + std * eps

    def marginal_x(self, x):
        x_recon, mu, logvar = self.forward(x)
        pdf = [scipy.stats.multivariate_normal.pdf(x, mu, 0.01)
               for x, mu in zip(x.data.numpy(), x_recon.data.numpy())]
        return np.array(pdf)


def KL_Gaussians(mu, logvar):
    # prior     p(z) = N(0,1)
    # posterior q(z) = N(mu, exp(logvar))
    # returns KL[ q(z) | p(z) ]
    return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class StandardNormal():
    def __init__(self, size):
        self.size = size

    def sample(self):
        return Variable(torch.randn(self.size))


class ReparamNormal(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=10, output_dim=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        self.fc_logvar = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        self.mu = 0
        self.logvar = 0

    def condition(self, x):
        self.mu, self.logvar = self.MLP(x)
        self.var = self.logvar.exp()
        self.std = self.var.sqrt()
        self.precision = 1.0/self.var

    def MLP(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def sample(self):
        """
        eps ~ N[0,1]
        std = sqrt(exp(logvar))
        z = mu + std * eps
        """
        eps = Variable(torch.randn(self.mu.size()))
        z = eps.mul(self.std).add(self.mu)
        return z

    def logpdf(self, x):
        """
        z ~ p(z|x)
        mu, var = MLP(z)
        log p(x|z) = -(x - mu)^2 / (2*var) - log(std) + C
        """
        lp = (x - self.mu).pow(2).mul(-0.5/self.var).add(-self.std.log())
        return torch.sum(lp, 1)

    def logInvGamma(self):
        shape = 2
        scale_std = 0.01
        scale = 1 / np.sqrt(scale_std)
        loggamma = (shape-1)*self.precision.log() - self.precision / scale

    def forward(self, x):
        return self.logpdf(x)

    def KL_StdNormal(self):
        '''
        prior     p(z) = N(0,1)
        posterior q(z) = N(mu, var * I)
        returns KL[ q(z) | p(z) ]
        '''
        kl_j = 1 + self.logvar - self.mu.pow(2) - self.var
        return -0.5*torch.sum(kl_j, 1)


class VAE(torch.nn.Module):
    def __init__(self, latent_dim=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.px = ReparamNormal(latent_dim, 100, 2)
        self.qz = ReparamNormal(2, 100, latent_dim)
        self.num_samples = 1

    def forward(self, x):
        elbo = self.ELBO(x)
        return torch.sum(elbo)

    def ELBO(self, x):
        """
        L = E[p(x|z)] - KL[q(z)|p(z)]
        """
        self.qz.condition(x)
        lp = self.Expectation(x, self.qz)
        kl = self.qz.KL_StdNormal()
        elbo = lp - kl
        return elbo

    def Expectation(self, x, qz):
        """
        z_l ~ q(z|x)
        mu, var = MLP(z_l)
        log p(x|z) = N(x; mu, var)
        E[ log p(x|z_l) ]
        """
        lp = 0
        for _ in range(self.num_samples):
            z = qz.sample()
            self.px.condition(z)
            lp += self.px.logpdf(x)
        return lp / self.num_samples

