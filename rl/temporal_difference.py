
learning_rate = 0.01
discount = 1.0


def TD_zero_update(value, s0, s1, reward):
    TD_target = reward + discount*value[s1]
    TD_error = TD_target - value[s0]
    value[s0] = value[s0] + learning_rate * TD_error


def TD_zero_evaluation(mdp, value, initial_state, MAX_ITER=10000):
    for iter in range(MAX_ITER):
        for (s0, a, r, s1) in mdp.Simulate(initial_state):
            TD_zero_update(value, s0, s1, r)

def SARSA( s0, a0, reward, s1, a1):
    TD_target = reward + discount*Q_values(s1,a1)
    TD_error = TD_target - Q_values(s0,a0)
    Q = Q_values(s0,a0) + learning_rate * TD_error
    Set_Q_values( Q)


def TD_zero_control():
    a0 = policy(s0)
    for iter in range(MAX_ITER):
       s1, reward = SampleNextState(s0, a0)
       a1 = policy(s1) 
       SARSA( Q_values, s0, a0, reward, s1, a1)
       s0 = s1
       a0 = a1


