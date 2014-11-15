# http://dogdogfish.com/2014/08/22/bayesian-testing-of-conversion-rate/

import random
import numpy as np

total_points = 50

trials = [random.randint(20,100) for _ in range(total_points)]
results = [
        np.random.binomial(value, 0.4) 
    if total_points/2 else 
        np.random.binomial(value, 0.3) 
    for index, value in enumerate(trials) ]

for trial, result in zip( trials, results):
    print "%d\t%d" % (trial, result)

# at the shell:
# $ python generate_conversions.py > conversion_data.txt
