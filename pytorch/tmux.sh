#!/usr/bin/env bash

NAME=pytorch

if tmux has-session -t $NAME; then
  tmux kill-session -t $NAME
else
  tmux new-session -A -s $NAME \; \
    send-keys 'R -q' C-m \; \
    rename-window 'R' \; \
    new-window -n 'julia' \; \
    send-keys 'while true;do julia;sleep 1;done' C-m \; \
    new-window -n 'ipython' \; \
    send-keys 'jupyter-console --existing' C-m \; \
    new-window -n 'sqlite' \; \
    send-keys 'sqlite3 data.db' \; \
    new-window -n 'bash' \; \
    send-keys 'jupyter-notebook' C-m \; \
    select-window -t ipython
fi

