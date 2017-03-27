let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-pytorch
  au!
" exe b:auBufEnter . '/app.py let b:tmux_window="web"'
  exe b:auBufEnter . '/* call functions#SwitchTmux("pytorch")'
augroup END

let g:slime_python_ipython = 1

augroup lint-on-change
  au!
  " autocmd BufWrite * Neomake
  autocmd InsertLeave,TextChanged *.py update|Neomake
augroup END

