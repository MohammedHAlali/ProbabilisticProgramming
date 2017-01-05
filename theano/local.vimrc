let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-theano
  au!
" exe b:auBufEnter . '/app.py let b:tmux_window="web"'
  exe b:auBufEnter . '/* call functions#SwitchTmux("theano")'
augroup END

let g:slime_python_ipython = 1

