import numpy as np
import torch

x = torch.Tensor(5,3)

x = torch.rand(5,3)

x

x.size()

y = torch.rand(5,3)

x + y

torch.add(x,y)

result = torch.Tensor(5,3)

torch.add(x, y, out=result)

y.add_(x)

a = torch.ones(5)

b = a.numpy()


a = np.ones(5)
b = torch.from_numpy(a)
np.add( a, 1, out=a)
print(a)
print(b)

torch.cuda.is_available()



