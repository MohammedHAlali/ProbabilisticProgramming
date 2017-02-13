# https://jeremykun.com/2016/02/08/big-dimensions-and-what-you-can-do-about-it/

import random
 
def randUnitCube(n):
   return [(random.random() - 0.5)*2 for _ in range(n)]
 
def sphereCubeRatio(n, numSamples):
   randomSample = [randUnitCube(n) for _ in range(numSamples)]
   return sum(1 for x in randomSample if sum(a**2 for a in x) >= 1) / numSamples


