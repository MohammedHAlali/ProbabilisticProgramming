import numpy as np

p_h = 0.4
discount = 1.0
MAXCAP = 100

ALL_STATES = np.arange(0,MAXCAP+1)
STATES = np.arange(1,MAXCAP) # excludes terminal states {0, MAXCAP}

ACTIONS = [np.arange(0, min(s0, MAXCAP-s0)+1) for s0 in ALL_STATES]

def ProbabilityTransition( s1, s0, action ):
    if s0 == MAXCAP or s0 == 0:
        return 0.0
    if s1 == s0 + action:
        return p_h
    if s1 == s0 - action:
        return 1-p_h
    return 0.0

def Reward(state):
    if state == MAXCAP:
        return 1
    return 0

def NextStates( s0, action):
    if action == 0:
        return [s0]
    if s0 == MAXCAP or s0 == 0:
        return []
    return s0 + action * np.array([-1, 1])

def ExpectedReturn(s0, action, value):
    v = 0.0
    for s1 in NextStates(s0, action):
        p = ProbabilityTransition(s1, s0, action)
        v += p * (Reward(s1) + discount * value[s1])
    return v

def ValueIteration( value_init, MAX_ITER=1000, eps=1e-8 ):
    value0 = value_init.copy()
    value1 = np.zeros_like(value0)
    for iter in range(MAX_ITER):
        max_diff = 0.0
        for s0 in STATES:
            actions = ACTIONS[s0]
            Er = [ExpectedReturn(s0, a, value0) for a in actions]
            value1[s0] = np.max( Er )
            max_diff = np.maximum(max_diff, np.abs(value1[s0]-value0[s0]))
        value0[:] = value1
        if max_diff < eps:
            return value1, iter
    return value1, iter

def GetPolicy( value ):
    policy = np.zeros(101,dtype=np.int)
    for s0 in STATES:
        actions = ACTIONS[s0]
        Er = [ExpectedReturn(s0, a, value) for a in actions]
        policy[s0] = np.argmax( Er )
    return policy

def GetQValue( value ):
    Qvalue = []
    for s0 in ALL_STATES:
        actions = ACTIONS[s0]
        Er = [ExpectedReturn(s0, a, value) for a in actions]
        Qvalue.append(Er)
    return Qvalue

def SimulateNextState(state, action):
    heads = np.random.binomial(1, p_h)
    heads = 2*heads - 1
    return state + heads * action

def Simulate(initial_state, policy, MAX_ITER=10000):
    s0 = initial_state
    history = [s0]
    for _ in range(MAX_ITER):
        action = policy[s0]
        s1 = SimulateNextState(s0, action)
        history.append(s1)
        if s1 == 0 or s1 == MAXCAP:
            break
        s0 = s1
    return history
