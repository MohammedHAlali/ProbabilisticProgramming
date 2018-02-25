#!/usr/bin/env bash

NAME=pyro

if tmux has-session -t $NAME; then
  tmux kill-session -t $NAME
else
  tmux new-session -A -s $NAME \; \
      send-keys 'jupyter-notebook' C-m \; \
      split-window -v -p80 \; \
    new-window -n 'ipython' \; \
      send-keys 'jupyter-console --existing' \; \
    select-window -t ipython
fi

