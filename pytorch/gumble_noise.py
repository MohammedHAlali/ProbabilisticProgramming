# https://github.com/pytorch/pytorch/issues/639

def sampler(input, tau, temperature):
    noise = torch.rand(input.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = Variable(noise)
    x = (input + noise) / tau + temperature
    x = F.softmax(x.view(input.size(0), -1))
    return x.view_as(input)
