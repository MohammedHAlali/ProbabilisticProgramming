import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import scipy
import numpy as np

# old model {{{
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
# }}}


LOG_SQRT_2PI = np.log(np.sqrt(2*np.pi))


class ReparamNormal(torch.nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        self.mu = 0
        self.logvar = 0

    def condition(self, x):
        self.mu, self.logvar = self.forward(x)
        self.var = self.logvar.exp()
        self.std = self.var.sqrt()

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
        mu = self.mu.expand_as(x)
        var = self.var.expand_as(x)
        std = self.std.expand_as(x)
        lp = (x - mu).pow(2).mul(-0.5/var).add(-std.log()).add(LOG_SQRT_2PI)
        return torch.sum(lp, 1)

    def KL_StdNormal(self):
        '''
        prior     p(z) = N(0,1)
        posterior q(z) = N(mu, var * I)
        returns KL[ q(z) | p(z) ]
        '''
        kl_j = 1 + self.logvar - self.mu.pow(2) - self.var
        return -0.5*torch.sum(kl_j, 1)


class ReparamNormal_Mu_Logvar(ReparamNormal):
    def __init__(self, output_dim=2, **kwargs):
        super().__init__()
        mu = kwargs.get('mu', torch.zeros(output_dim))
        self.mu_param = Parameter(mu.unsqueeze(0))
        logvar = kwargs.get('logvar', torch.zeros(output_dim))
        self.logvar_param = Parameter(logvar.unsqueeze(0))

    def forward(self, x):
        s = x.size(0)
        o = self.mu_param.size(1)
        return self.mu_param.expand(s, o), self.logvar_param.expand(s, o)

    def __repr__(self):
        return super().__repr__() + \
            '(_ -> {})'.format(self.mu_param.size(0))


class BNAffine(torch.nn.Module):
    def __init__(self, input_dim, output_dim, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.fc = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.batchnorm = torch.nn.BatchNorm1d(output_dim, affine=False)
        self.affine = torch.nn.Linear(output_dim, output_dim)
        self.affine.weight.data = (torch.eye(output_dim) +
                                   eps*torch.randn(output_dim, output_dim))
        self.affine.bias.data.uniform_(-eps, eps)

    def forward(self, h):
        x = self.fc(h)
        x = self.batchnorm(x)
        x = self.affine(x)
        return x

    def init_scale(self, scale):
        scale = torch.Tensor(scale)
        self.affine.weight.data = torch.diag(scale)

    def init_shift(self, shift):
        self.affine.bias.data = torch.Tensor(shift)


class ReparamNormal_MLP(ReparamNormal):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=10):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn1.weight.data = self.bn1.weight.data*1e-5 + 0.1
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bna_mu = BNAffine(hidden_dim, output_dim)
        self.bna_logvar = BNAffine(hidden_dim, output_dim)

        eps = 1e-5
        self.fc1.bias.data.uniform_(-eps, eps)
        self.fc2.bias.data.uniform_(-eps, eps)

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.tanh(h)
        mu = self.bna_mu(h)
        logvar = self.bna_logvar(h)
        return mu, logvar


class ReparamNormal_MLP_Logvar(ReparamNormal):
    def __init__(self, input_dim=2, output_dim=2, hidden_dim=10, **kwargs):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        logvar = kwargs.get('logvar', torch.zeros(output_dim))
        self.logvar_param = Parameter(logvar.unsqueeze(0))

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        return mu, self.logvar_param.expand_as(mu)


class Maxout(torch.nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.fc = torch.nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        last_dim = len(shape) - 1
        out = self.fc(inputs)
        m, i = out.view(*shape).max(last_dim)
        return m.squeeze(last_dim)


class VAE(torch.nn.Module):  # {{{
    def __init__(self, latent_dim=1, obs_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.px = ReparamNormal(latent_dim, 100, obs_dim)
        self.qz = ReparamNormal(obs_dim, 100, latent_dim)
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
# }}}


class AuxiliaryDGM(torch.nn.Module):
    def __init__(self, num_samples=10, aux_dim=1, latent_dim=2):
        """
        q(a) = N(0,I)
        q(z|a) = N(z; mu(a), var(a))
        p(a|z) = N(a; mu(z), var(z))
        """
        super().__init__()
        self.num_samples = num_samples
        self.aux_dim = aux_dim
        self.latent_dim = latent_dim
        self.annealing = 0.0
        self.qa = StandardNormal((num_samples, aux_dim))
        self.qz = None
        self.pa = None

    def sample(self):
        return self.sample_inference()

    def sample_inference(self):
        """
        q(a,z|x) = q(z|a,x) q(a)
        pa = p(a|z)
        """
        aux_samples = self.qa.sample()
        self.qz.condition(aux_samples)
        z = self.qz.sample()
        self.pa.condition(z)
        self.z = z
        self.aux_samples = aux_samples
        return (z, aux_samples)

    def ELBO(self):
        '''
        p(x|z) * p(a|z) / q(z|a) * q(a)
        E_q(a,z)
        '''
        z, aux_samples = self.sample_inference()
        elbo = (self.px_logpdf(z)
                + self.pa.logpdf(aux_samples)
                - self.qa.logpdf(aux_samples)
                - self.qz.logpdf(z))
        return torch.sum(elbo) / self.num_samples

    def px_logpdf(self, z):
        return 10**-self.annealing * self.generative(z)

    def generative(self, z):
        return 0


class AuxiliaryDGM_(AuxiliaryDGM):
    def __init__(self, num_samples=10, aux_dim=1, latent_dim=2):
        super().__init__(num_samples, aux_dim, latent_dim)
        self.qz = ReparamNormal(aux_dim, 10, latent_dim)
        self.pa = ReparamNormal(latent_dim, 10, aux_dim)


class BarberExample2(AuxiliaryDGM):
    def __init__(self, num_samples, aux_dim):
        super().__init__(num_samples, aux_dim)
        latent_dim = 2
        self.qz = ReparamNormal_MLP(input_dim=aux_dim,
                                    output_dim=latent_dim,
                                    hidden_dim=20)
        self.qz.bna_mu.init_shift([0, 0])
        self.qz.bna_mu.init_scale([4, 4])
        self.qz.bna_logvar.init_shift([-4, -4])
        self.qz.bna_logvar.init_scale([.1, .1])
        self.pa = ReparamNormal_MLP(latent_dim, aux_dim)
        self.pa.bna_logvar.init_shift(-3.*np.ones(aux_dim))
        self.pa.bna_logvar.init_scale(-0.1*np.ones(aux_dim))
        # self.pa = ReparamNormal_Mu_Logvar(output_dim=aux_dim)
        self.ig = InverseGamma(0.4, 100)
        self.ndist = NormalDist()
        self.y2 = -6
        self.q = 1

    def sample(self):
        z, aux = super().sample()
        mu, logvar = torch.unbind(z, 1)
        return (mu.data.numpy(), logvar.data.numpy())

    def generative(self, z):
        mu, logvar = torch.unbind(z, 1)
        var = logvar.exp()
        lp = (self.ndist.logpdf(self.y2, mu, var)
              + self.ndist.logpdf(mu, 0, self.q)
              + self.ig.logpdf(var))
        return lp


class PotentialModel(AuxiliaryDGM):
    def __init__(self):
        super().__init__()

    def logpdf(z):
        norm = torch.norm(z)
        circle = 0.5 * ((norm - 2) / 0.4)**2
        half1 = -0.5 * ((z[0] - 2) / 0.6)**2
        half2 = -0.5 * ((z[0] + 2) / 0.6)**2
        return circle - torch.log(half1.exp() + half2.exp())


class DeepMLP(torch.nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.num_samples = 100
        self.aux_dim = 20
        self.hidden_dim = 20
        self.output_dim = 2
        self.fc = []
        self.fc.append(Linear(self.aux_dim, self.hidden_dim))
        for i in range(depth):
            self.fc.append(LinearMPELU(self.hidden_dim, self.hidden_dim))
        self.fc.append(LinearMPELU(self.hidden_dim, self.output_dim))

    def forward(self, x):
        h = self.fc[0](x)
        self.h = [x, h]
        D = len(self.fc)
        for i in range(1, D):
            h = F.elu(h)
            h = self.fc[i](h)
            self.h.append(h)
        return h

    def sample(self):
        s = Variable(torch.randn((self.num_samples, self.aux_dim)))
        return self.forward(s)


class Linear(torch.nn.Linear):
    def __init__(self, inputSize, outputSize, bias=True):
        super().__init__(inputSize, outputSize, bias)
        self.reset()

    def reset(self, stdv=None):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        self.bias.data.zero_()


class LinearMPELU(torch.nn.Linear):
    def __init__(self, inputSize, outputSize, bias=True):
        super().__init__(inputSize, outputSize, bias)
        self.reset()

    def reset(self, stdv=None):
        c = self.weight.size(1)
        k = 1.0
        alpha = 0.4
        beta = 1.0

        d = k**2 * c * (1 + alpha**2 * beta**2)
        stdv = np.sqrt(2/d)

        print('stdv: {}'.format(stdv))
        self.weight.data.normal_(0, stdv)

        if self.bias is not None:
            self.bias.data.zero_()

        return self


# Distributions {{{


class NormalDist():
    def logpdf(self, x, mu, var):
        if type(var) is Variable:
            std = var.sqrt()
            logstd = std.log()
        else:
            std = np.sqrt(var)
            logstd = np.log(std)
        lp = (x - mu).pow(2).mul(-0.5/var).add(-logstd).add(-LOG_SQRT_2PI)
        return lp


class InverseGamma():
    def __init__(self, v, beta):
        self.v = v
        self.beta = beta

    def logpdf(self, r):
        v = self.v
        beta = self.beta
        if type(r) is Variable:
            logr = r.log()
        else:
            logr = np.log(r)
        return -(v+1)*logr - v/(beta * r)


class StandardNormal():
    def __init__(self, size):
        self.size = size

    def sample(self):
        return Variable(torch.randn(self.size))

    def logpdf(self, x):
        lp = x.pow(2).mul(-0.5).add(-LOG_SQRT_2PI)
        return torch.sum(lp, 1)
# }}}
