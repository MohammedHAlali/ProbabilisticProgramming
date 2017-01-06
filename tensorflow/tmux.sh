#!/usr/bin/env bash

NAME=tensorflow

if tmux has-session -t $NAME; then
  tmux kill-session -t $NAME
else
  tmux new-session -A -s $NAME \; \
    new-window -n 'ipython' \; \
    send-keys 'ipython2 --no-banner' C-m \; \
    select-window -t 2
fi

