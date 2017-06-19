# http://dogdogfish.com/2014/08/22/bayesian-testing-of-conversion-rate/

import numpy  as np
import matplotlib.pyplot as plt
import pymc as pm


basket_list = []
conversion_list = []

with open( 'conversion_data.txt', 'rb' ) as f:
    for line in f:
        baskets, conversions = line.strip().split()
        basket_list.append( int(baskets) )
        conversion_list.append( int( conversions ) )

n_percent_list = len( basket_list )

uniform_one_samples = []
uniform_two_samples = []
tau_samples = []
uniform_one = pm.Uniform( 'uniform_one', 0, 1 )
uniform_two = pm.Uniform( 'uniform_two', 0, 1 )

tau = pm.Discre





