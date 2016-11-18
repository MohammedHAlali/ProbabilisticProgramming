#!/usr/bin/env bash

NAME=mxnet

if tmux has-session -t $NAME; then
  tmux kill-window -a -t $NAME:R
else
  tmux new-session -A -s $NAME \; \
    send-keys 'R -q' C-m \; \
    rename-window 'R' \; \
    new-window -n 'julia' \; \
    send-keys 'while true;do julia;sleep 1;done' C-m \; \
    new-window -n 'ipython' \; \
    send-keys 'ipython --no-banner' C-m \; \
    new-window -n 'sqlite' \; \
    send-keys 'sqlite3 data.db' \; \
    new-window -n 'bash' \; \
    send-keys 'bash' C-m \; \
    select-window -t 3
fi

