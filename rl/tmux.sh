#!/usr/bin/env bash

NAME=rl

if tmux has-session -t $NAME; then
  tmux kill-session -t $NAME
else
  tmux new-session -A -s $NAME \; \
    send-keys 'R -q' C-m \; \
    rename-window 'R' \; \
    new-window -n 'julia' \; \
    send-keys 'while true;do julia;sleep 1;done' C-m \; \
    new-window -n 'ipython' \; \
    send-keys 'ipython3 --no-banner' C-m \; \
    new-window -n 'sqlite' \; \
    send-keys 'sqlite3 data.db' \; \
    new-window -n 'bash' \; \
    send-keys 'bash' C-m \; \
    select-window -t 2
fi

