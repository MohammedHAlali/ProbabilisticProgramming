let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-mxnet
  au!
  exe b:auBufEnter . '/*.R   silent !tmux select-window -t mxnet:R'
  exe b:auBufEnter . '/*.jl  silent !tmux select-window -t mxnet:julia'
  exe b:auBufEnter . '/*.py  silent !tmux select-window -t mxnet:ipython'
  exe b:auBufEnter . '/*.sql silent !tmux select-window -t mxnet:sqlite'
  exe b:auBufEnter . '/*.sh  silent !tmux select-window -t mxnet:bash'
  exe b:auBufEnter . '/*.R   let b:slime_config = {"socket_name": "default", "target_pane": "mxnet:R"}'
  exe b:auBufEnter . '/*.jl  let b:slime_config = {"socket_name": "default", "target_pane": "mxnet:julia"}'
  exe b:auBufEnter . '/*.py  let b:slime_config = {"socket_name": "default", "target_pane": "mxnet:ipython"}'
  exe b:auBufEnter . '/*.sql let b:slime_config = {"socket_name": "default", "target_pane": "mxnet:sqlite"}'
  exe b:auBufEnter . '/*.sh  let b:slime_config = {"socket_name": "default", "target_pane": "mxnet:bash"}'
augroup END

let g:slime_python_ipython = 1
