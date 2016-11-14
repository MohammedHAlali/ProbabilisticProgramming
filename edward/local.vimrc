let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-edward
  au!
  exe b:auBufEnter . '/*.R   silent !tmux select-window -t edward:R'
  exe b:auBufEnter . '/*.jl  silent !tmux select-window -t edward:julia'
  exe b:auBufEnter . '/*.py  silent !tmux select-window -t edward:ipython'
  exe b:auBufEnter . '/*.sql silent !tmux select-window -t edward:sqlite'
  exe b:auBufEnter . '/*.sh  silent !tmux select-window -t edward:bash'
  exe b:auBufEnter . '/*.R   let b:slime_config = {"socket_name": "default", "target_pane": "edward:R"}'
  exe b:auBufEnter . '/*.jl  let b:slime_config = {"socket_name": "default", "target_pane": "edward:julia"}'
  exe b:auBufEnter . '/*.py  let b:slime_config = {"socket_name": "default", "target_pane": "edward:ipython"}'
  exe b:auBufEnter . '/*.sql let b:slime_config = {"socket_name": "default", "target_pane": "edward:sqlite"}'
  exe b:auBufEnter . '/*.sh  let b:slime_config = {"socket_name": "default", "target_pane": "edward:bash"}'
augroup END

let g:slime_python_ipython = 1
