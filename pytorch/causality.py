# http://www.homepages.ucl.ac.uk/~ucgtrbd/papers/causality.pdf
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

a = torch.randn(1)
b = torch.randn(1)
c = torch.randn(1)
print( (a,b,c) )

hist = []
for t in range(1000):
    epsX = torch.randn(1)
    epsY = torch.randn(1)
    H = torch.randn(1)
    X = a * H + epsX
    Y = c * H + b * X + epsY
    hist.append((H[0],X[0],Y[0]))
hist = np.array(hist)
H, X, Y = hist.T
plt.clf()
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.scatter(X,Y)

hist = []
for X in np.arange(-3,3,.1):
    for t in range(100):
        epsX = torch.randn(1)
        epsY = torch.randn(1)
        H = torch.randn(1)
        Y = c * H + b * X + epsY
        hist.append((H[0],X,Y[0]))
hist = np.array(hist)
H, X, Y = hist.T
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.scatter(X,Y)


hist = []
for t in range(1000):
    epsX = torch.randn(1)
    epsY = torch.randn(1)
    H = torch.randn(1)
    Z = H
    X = a * Z + epsX
    Y = c * H + b * X + epsY
    hist.append((H[0],Z[0],X[0],Y[0]))
hist = np.array(hist)
H, Z, X, Y = hist.T
plt.clf()
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.scatter(X,Y)
