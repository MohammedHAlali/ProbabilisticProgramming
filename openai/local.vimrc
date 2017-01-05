let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-openai
  au!
  exe b:auBufEnter . '/* call functions#SwitchTmux("openai")'
augroup END

let g:slime_python_ipython = 1
