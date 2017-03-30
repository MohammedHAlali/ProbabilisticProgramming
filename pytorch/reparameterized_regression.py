import torch
import torch.nn.functional as F
from torch.autograd import Variable


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


