#!/usr/bin/env bash

VENV="source ./venv/bin/activate"

NAME=tensorflow

if tmux has-session -t $NAME; then
  tmux kill-session -t $NAME
else
  tmux new-session -A -s $NAME \; \
    send-keys "$VENV" C-m \; \
    new-window -n 'ipython' \; \
    send-keys "$VENV;ipython" C-m \; \
    select-window -t 2
fi

