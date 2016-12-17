let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-openai
  au!
  exe b:auBufEnter . '/*.R   silent !tmux select-window -t openai:R'
  exe b:auBufEnter . '/*.jl  silent !tmux select-window -t openai:julia'
  exe b:auBufEnter . '/*.py  silent !tmux select-window -t openai:ipython'
  exe b:auBufEnter . '/*.sql silent !tmux select-window -t openai:sqlite'
  exe b:auBufEnter . '/*.sh  silent !tmux select-window -t openai:bash'
  exe b:auBufEnter . '/*.R   let b:slime_config = {"socket_name": "default", "target_pane": "openai:R"}'
  exe b:auBufEnter . '/*.jl  let b:slime_config = {"socket_name": "default", "target_pane": "openai:julia"}'
  exe b:auBufEnter . '/*.py  let b:slime_config = {"socket_name": "default", "target_pane": "openai:ipython"}'
  exe b:auBufEnter . '/*.sql let b:slime_config = {"socket_name": "default", "target_pane": "openai:sqlite"}'
  exe b:auBufEnter . '/*.sh  let b:slime_config = {"socket_name": "default", "target_pane": "openai:bash"}'
augroup END

let g:slime_python_ipython = 1
