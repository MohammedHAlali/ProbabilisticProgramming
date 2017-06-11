import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch import FloatTensor


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)
        self.obs_var = 1

    def forward(self, X):
        return self.simulate(X)

    def simulate(self, X):
        Y = self.l1(X)
        noise = self.obs_var * torch.randn(Y.size())
        noise = Variable(noise)
        Y.add_(noise)
        return Y


class BayesianModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b_param = RepNormal()
        self.w_param = RepNormal()

    def forward(self, X):
        return self.simulate(X)

    def simulate(self, X):
        b = self.b_param().expand(X.size())
        w = self.w_param().unsqueeze(0)
        Y = b + X.mm(w) + Variable(torch.randn(b.size()))
        return Y


class RepNormal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = Parameter(FloatTensor([0.0]))
        self.log_variance = Parameter(FloatTensor([0.0]))

    def __call__(self):
        z = Variable(torch.randn(1))
        return self.mu + self.log_variance.exp() * z

    def _repr_pretty_(self, p, cycle):
        p.text("mu = {}".format(self.mu))
        p.text("std = {}".format(self.log_variance.exp()))

class Poster
