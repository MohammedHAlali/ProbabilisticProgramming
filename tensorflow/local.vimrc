let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-tensorflow
  au!
" exe b:auBufEnter . '/app.py let b:tmux_window="web"'
  exe b:auBufEnter . '/* call functions#SwitchTmux("tensorflow")'
augroup END

let g:slime_python_ipython = 1
