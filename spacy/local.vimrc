let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-spacy
  au!
  exe b:auBufEnter . '/*.R   silent !tmux select-window -t spacy:R'
  exe b:auBufEnter . '/*.jl  silent !tmux select-window -t spacy:julia'
  exe b:auBufEnter . '/*.py  silent !tmux select-window -t spacy:ipython'
  exe b:auBufEnter . '/*.sql silent !tmux select-window -t spacy:sqlite'
  exe b:auBufEnter . '/*.sh  silent !tmux select-window -t spacy:bash'
  exe b:auBufEnter . '/*.R   let b:slime_config = {"socket_name": "default", "target_pane": "spacy:R"}'
  exe b:auBufEnter . '/*.jl  let b:slime_config = {"socket_name": "default", "target_pane": "spacy:julia"}'
  exe b:auBufEnter . '/*.py  let b:slime_config = {"socket_name": "default", "target_pane": "spacy:ipython"}'
  exe b:auBufEnter . '/*.sql let b:slime_config = {"socket_name": "default", "target_pane": "spacy:sqlite"}'
  exe b:auBufEnter . '/*.sh  let b:slime_config = {"socket_name": "default", "target_pane": "spacy:bash"}'
augroup ENDo

let g:slime_python_ipython = 1

