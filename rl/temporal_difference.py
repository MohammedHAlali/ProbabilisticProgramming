
learning_rate = 0.01
discount = 1.0


def TD_zero_evaluation(mdp, value, initial_state, MAX_ITER=10000):
    for iter in range(MAX_ITER):
        for (s0, a0, r, s1, a1) in mdp.Simulate(initial_state):
            TD_target = r + discount*value[s1]
            TD_error = TD_target - value[s0]
            value[s0] = value[s0] + learning_rate * TD_error


def TD_zero_control(mdp, initial_state, MAX_ITER=1):
    for iter in range(MAX_ITER):
        for (s0, a0, r, s1, a1) in mdp.Simulate(initial_state):
            Q1 = mdp.GetQ(s1, a1)
            Q0 = mdp.GetQ(s0, a0)
            TD_target = r + discount*Q1
            TD_error = TD_target - Q0
            Q0 = Q0 + learning_rate * TD_error
            mdp.SetQ(s0, a0, Q0)
