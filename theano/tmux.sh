#!/usr/bin/env bash

NAME=theano

if tmux has-session -t $NAME; then
  tmux kill-session -t $NAME
else
  tmux new-session -A -s $NAME \; \
    new-window -n 'ipython' \; \
    send-keys 'ipython3 --no-banner' C-m \; \
    select-window -t 1
fi

