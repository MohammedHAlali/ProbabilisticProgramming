# http://pyro.ai/examples/intro_part_i.html

import torch
from torch.autograd import Variable
import pyro
import pyro.distributions as dist

mu = Variable(torch.zeros(1))
sigma = Variable(torch.ones(1))
x = dist.normal(mu, sigma)
print(x)

log_p_x = dist.normal.log_pdf(x, mu, sigma)
print(log_p_x)

x = pyro.sample('my_sample', dist.normal, mu, sigma)
print(x)


def weather():
    cloudy = pyro.sample('cloudy', dist.bernoulli,
                         Variable(torch.Tensor([0.3])))
    cloudy = 'cloudy' if cloudy.data[0] == 1.0 else 'sunny'
    mean_temp = {'cloudy': [55.0], 
                 'sunny': [75.0]}[cloudy]
    sigma_temp = {'cloudy': [10.0],
                  'sunny': [15.0]}[cloudy]
    temp = pyro.sample('temp', dist.normal, Variable(torch.Tensor(mean_temp)),
                       Variable(torch.Tensor(sigma_temp)))
    return cloudy, temp.data[0]

for _ in range(3):
    print(weather())

def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = [200] if cloudy == 'sunny' and temp > 80.0 else [50]
    ice_cream = pyro.sample('ice_cream', dist.normal,
                            Variable(torch.Tensor(expected_sales)),
                            Variable(torch.Tensor([10.0])))
    return ice_cream

ice_cream_sales()


def geometric(p, t=None):
    t = 0 if t is None else t
    x = pyro.sample("x_{}".format(t), dist.bernoulli, p)
    if torch.equal(x.data, torch.zeros(1)):
        return x
    else:
        return x + geometric(p, t+1)

g = geometric(Variable(torch.Tensor([0.5])))

print(geometric(Variable(torch.Tensor([0.5]))))


def normal_product(mu, sigma):
    z1 = pyro.sample('z1', dist.normal, mu, sigma)
    z2 = pyro.sample('z2', dist.normal, mu, sigma)
    y = z1 * z2
    return y


def make_normal_normal():
    mu_latent = pyro.sample('mu_latent', dist.normal,
                            Variable(torch.zeros(1)),
                            Variable(torch.ones(1)))
    def fn(sigma):
        return normal_product(mu_latent, sigma)
    return fn


f = make_normal_normal()
f(Variable(torch.ones(1)))




