from value_iteration import *
from bandit import inc_avg

def SampleTrajectory(initial_state, policy, MAX_ITER=1000):
    s0 = initial_state
    history = []
    for _ in range(MAX_ITER):
        action = policy[s0]
        history.append([s0, action])
        s1 = SimulateNextState(s0, action)
        if s1 == 0 or s1 == MAXCAP:
            history.append([s1,None])
            break
        s0 = s1
    G = Reward(s1)
    return history, G

def SampleTrajectory_ES(initial_state, initial_action, policy, MAX_ITER=1000):
    s1 = SimulateNextState(initial_state, initial_action)
    history, G = SampleTrajectory( s1, policy, MAX_ITER)
    history.insert(0,[initial_state, initial_action])
    return history, G

def MonteCarloEvaluation(initial_state, policy, MAX_ITER=1000):
    v = np.zeros_like(ALL_STATES, dtype=np.float)
    count = np.zeros_like(ALL_STATES)
    for iter in range(MAX_ITER):
        traj, G = SampleTrajectory(initial_state, policy)
        for s in traj:
            count[s] += 1
            v[s] = inc_avg( G, v[s], count[s])
    return v

def MonteCarloControl_ES( Q_values, policy, MAX_ITER=10000):
    count = np.zeros_like(Q_values)
    for iter in range(MAX_ITER):
        s0 = np.random.choice(STATES)
        a0 = np.random.choice(ACTIONS[s0])
        history, G = SampleTrajectory_ES( s0, a0, policy )
        for s,a in history:
            if a == None:
                break
            count[s,a] += 1
            Q_values[s,a] = inc_avg( G, Q_values[s,a], count[s,a])
        for s in STATES:
            policy[s] = np.argmax( Q_values[s,ACTIONS[s]] )
    return policy

def MonteCarloControl( Q_values, MAX_ITER=10000 ):
    count = np.zeros_like(Q_values)
    for iter in range(MAX_ITER):
        traj, G = SampleTrajectory(SoftPolicy)
        for s in traj:
            for a in ACTIONS[s]:
                count[a,s] += 1
                Q_values[a,s] = inc_avg( G, Q_values[a,s], count[s])
    return Q_values


def eps_greedy(a_star, actions, eps=0.1):
    if np.random.rand() < eps:
        np.random.choice(actions)
    else:
        return a_star

def SoftPolicy(Q_value, state):
    a_star = np.argmax(Q_values(state))
    return eps_greedy( a_star, ACTIONS[state] )   

def init_Q():
    num_states = ALL_STATES.shape[0]
    num_actions = max([len(a) for a in ACTIONS]) 
    Q_values = np.zeros( (num_states, num_actions) )
    return Q_values

def init_policy():
    policy = [np.random.choice(actions) for actions in ACTIONS]
    return policy

