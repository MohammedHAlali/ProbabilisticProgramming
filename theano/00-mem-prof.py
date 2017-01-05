import copy

import memory_profiler

@profile
def function():
    x = list( range( 1000000 ) )
    y = copy.deepcopy( x )
    del x
    return y

if __name__ == "__main__":
    function()

# 11:43:59 AM $ python3 -m memory_profiler ./mem-prof.py
# Filename: ./mem-prof.py

# Line #    Mem usage    Increment   Line Contents
# ================================================
#      5   31.863 MiB    0.000 MiB   @profile
#      6                             def function():
#      7   70.535 MiB   38.672 MiB       x = list( range( 1000000 ) )
#      8   78.352 MiB    7.816 MiB       y = copy.deepcopy( x )
#      9   70.770 MiB   -7.582 MiB       del x
#     10   70.770 MiB    0.000 MiB       return y


