#!/usr/bin/env bash

NAME=lasagne

if tmux has-session -t $NAME; then
  tmux kill-window -a -t $NAME:R
else
  tmux new-session -A -s $NAME \; \
    new-window -n 'ipython' \; \
    send-keys 'ipython3 --no-banner' C-m \; \
    select-window -t 2
fi

