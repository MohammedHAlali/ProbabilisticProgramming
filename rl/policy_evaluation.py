import numpy as np

###############
#  .  1  2  3 #
#  4  5  6  7 #
#  8  9 10 11 #
# 12 13 14  . #
###############

discount = 1

T = 0
gridworld = np.array([
        [T, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, T]])

STATES = list(range(1,15))

value = np.zeros(16)

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
ACTIONS = [UP, DOWN, RIGHT, LEFT]
directions = np.array([
    [-1, 0],
    [ 1, 0],
    [0,1],
    [0,-1] 
])

def Next_State(state, action):
    loc = np.argwhere(gridworld==state)[0]
    p = loc + directions[action]
    if p[0] < 0:
        p[0] = 0
    if p[1] < 0:
        p[1] = 0
    if p[0] > 3:
        p[0] = 3
    if p[1] > 3:
        p[1] = 3
    return gridworld[p[0],p[1]]

def ProbabilityTransition( state, action):
    pass

def Reward(state, action):
    if state == T:
        return -1
    return -1

def PolicyProbability(action, state):
    return 0.25
    
def PolicyEvaluation(policy, old_value, MAX_ITER=1000, eps=1e-8):
    value = np.zeros_like(old_value)
    for iter in range(MAX_ITER):
        max_diff = 0.0
        value[:] = 0.0
        for s0 in STATES:
            for a in ACTIONS:
                pi = PolicyProbability(a,s0)
                s1 = Next_State(s0,a)
                value[s0] += pi * (Reward(s1,a) + discount*old_value[s1])
            max_diff = np.maximum(max_diff, np.abs(value[s0]-old_value[s0]))
            print(iter, max_diff)
        if max_diff < eps:
            return value, iter
        old_value[:] = value
    return value, iter
