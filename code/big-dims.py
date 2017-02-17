# https://jeremykun.com/2016/02/08/big-dimensions-and-what-you-can-do-about-it/
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()


import matplotlib

matplotlib.rcsetup.all_backends

matplotlib.get_backend()

matplotlib.matplotlib_fname()

 
def randUnitCube(n):
   return [(random.random() - 0.5)*2 for _ in range(n)]
 
def sphereCubeRatio(n, numSamples):
   randomSample = [randUnitCube(n) for _ in range(numSamples)]
   return sum(1 for x in randomSample if sum(a**2 for a in x) >= 1) / numSamples


dat = randUnitCube(3)
plt.clf()
plt.hist(dat)

sphereCubeRatio( 1, 3)

n = 3
numSamples = 100
randomSample = [randUnitCube(n) for _ in range(numSamples)]
[1 for x in randomSample if sum(a**2 for a in x) >= 1]

for x in randomSample:
    print( sum( a**2 for a in x))

randomSample[1]^2



n = 3
numSamples = 100000
randomSample = np.random.uniform(low=-1, high=1, size=(n, numSamples))
outside_sphere = np.sum(randomSample**2, axis=0) >= 1
1 - np.sum(outside_sphere.astype(int)) / numSamples

def sphereCubeRatio(n, numSamples):
    randomSample = np.random.uniform(low=-1, high=1, size=(n, numSamples))
    outside_sphere = np.sum(randomSample**2, axis=0) >= 1
    return 1 - np.sum(outside_sphere.astype(int)) / numSamples


dims = range(2, 20)
ratio = [ sphereCubeRatio(n, 1000000) for n in dims]

plt.clf()
plt.yscale('log')
plt.ylim(1e-10,1)
plt.scatter(dims,ratio)


import random
import math
import numpy
 
def randomSubspace(subspaceDimension, ambientDimension):
   return numpy.random.normal(0, 1, size=(subspaceDimension, ambientDimension))
 
def project(v, subspace):
   subspaceDimension = len(subspace)
   return (1 / math.sqrt(subspaceDimension)) * subspace.dot(v)
