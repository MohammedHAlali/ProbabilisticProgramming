#!/usr/bin/env bash

NAME=variational

if tmux has-session -t $NAME; then
  tmux kill-window -a -t $NAME:R
else
  tmux new-session -A -s $NAME \; \
    send-keys 'R -q' C-m \; \
    rename-window 'R' \; \
    new-window -n 'julia' \; \
    send-keys 'julia' C-m \; \
    select-window -t 2
fi

