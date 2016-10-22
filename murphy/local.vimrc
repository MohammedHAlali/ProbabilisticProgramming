let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-murphy
  au!
  exe b:auBufEnter . '/*.R   silent !tmux select-window -t murphy:R'
  exe b:auBufEnter . '/*.jl  silent !tmux select-window -t murphy:julia'
  exe b:auBufEnter . '/*.py  silent !tmux select-window -t murphy:ipython'
  exe b:auBufEnter . '/*.sql silent !tmux select-window -t murphy:sqlite'
  exe b:auBufEnter . '/*.R   let b:slime_config = {"socket_name": "default", "target_pane": "murphy:R"}'
  exe b:auBufEnter . '/*.jl  let b:slime_config = {"socket_name": "default", "target_pane": "murphy:julia"}'
  exe b:auBufEnter . '/*.py  let b:slime_config = {"socket_name": "default", "target_pane": "murphy:ipython"}'
  exe b:auBufEnter . '/*.sql let b:slime_config = {"socket_name": "default", "target_pane": "murphy:sqlite"}'
augroup END
