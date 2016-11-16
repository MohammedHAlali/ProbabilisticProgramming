let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-tensorflow
  au!
  exe b:auBufEnter . '/*.R   silent !tmux select-window -t tensorflow:R'
  exe b:auBufEnter . '/*.jl  silent !tmux select-window -t tensorflow:julia'
  exe b:auBufEnter . '/*.py  silent !tmux select-window -t tensorflow:ipython'
  exe b:auBufEnter . '/*.sql silent !tmux select-window -t tensorflow:sqlite'
  exe b:auBufEnter . '/*.sh  silent !tmux select-window -t tensorflow:bash'
  exe b:auBufEnter . '/*.R   let b:slime_config = {"socket_name": "default", "target_pane": "tensorflow:R"}'
  exe b:auBufEnter . '/*.jl  let b:slime_config = {"socket_name": "default", "target_pane": "tensorflow:julia"}'
  exe b:auBufEnter . '/*.py  let b:slime_config = {"socket_name": "default", "target_pane": "tensorflow:ipython"}'
  exe b:auBufEnter . '/*.sql let b:slime_config = {"socket_name": "default", "target_pane": "tensorflow:sqlite"}'
  exe b:auBufEnter . '/*.sh  let b:slime_config = {"socket_name": "default", "target_pane": "tensorflow:bash"}'
augroup END


let g:slime_python_ipython = 1
