let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-keras
  au!
  exe b:auBufEnter . '/*.R   silent !tmux select-window -t keras:R'
  exe b:auBufEnter . '/*.jl  silent !tmux select-window -t keras:julia'
  exe b:auBufEnter . '/*.py  silent !tmux select-window -t keras:ipython'
  exe b:auBufEnter . '/*.sql silent !tmux select-window -t keras:sqlite'
  exe b:auBufEnter . '/*.sh  silent !tmux select-window -t keras:bash'
  exe b:auBufEnter . '/*.R   let b:slime_config = {"socket_name": "default", "target_pane": "keras:R"}'
  exe b:auBufEnter . '/*.jl  let b:slime_config = {"socket_name": "default", "target_pane": "keras:julia"}'
  exe b:auBufEnter . '/*.py  let b:slime_config = {"socket_name": "default", "target_pane": "keras:ipython"}'
  exe b:auBufEnter . '/*.sql let b:slime_config = {"socket_name": "default", "target_pane": "keras:sqlite"}'
  exe b:auBufEnter . '/*.sh  let b:slime_config = {"socket_name": "default", "target_pane": "keras:bash"}'
augroup END

let g:slime_python_ipython = 1
