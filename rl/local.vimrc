let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-rl
  au!
" exe b:auBufEnter . '/app.py let b:tmux_window="web"'
  exe b:auBufEnter . '/* call functions#SwitchTmux("rl")'
augroup END

let g:slime_python_ipython = 1

augroup lint-on-change
  au!
  " autocmd BufWrite * Neomake
  autocmd InsertLeave,TextChanged * update|Neomake
augroup END

let s:efm  = '%G<ipython-%.%#,'
let s:efm .= '%E%f in %.%#(%.%#),'
let s:efm .= '%C---> %l %.%#,'
let s:efm .= '%C     %.%#,'
let s:efm .= '%C,'
let s:efm .= '%Z%m'

let &l:efm = s:efm
