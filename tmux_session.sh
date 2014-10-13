tmux new-session -A -s probprog \; \
    send-keys 'ipython' C-m \; \
    new-window -n 'server' \; \
    send-keys 'cd ~/apps/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers; ipython notebook' C-m \; \
    new-window -n 'pydoc' \; \
    send-keys 'pydoc -p 8080'
    select-window -t 1
