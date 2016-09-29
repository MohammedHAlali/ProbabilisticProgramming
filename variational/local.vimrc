let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-variational
	au!
	exe b:auBufEnter . '/*.R   silent !tmux select-window -t variational:julia'
  exe b:auBufEnter . '/*.R   let b:slime_config = {"socket_name": "default", "target_pane": "variational:julia"}'
	exe b:auBufEnter . '/*.jl  silent !tmux select-window -t variational:julia'
	exe b:auBufEnter . '/*.jl  let b:slime_config = {"socket_name": "default", "target_pane": "variational:julia"}'
augroup END

" RCall R-block in julia REPL
function! _EscapeText_r(text)
  return ["R\"\"\"\n", a:text."\"\"\"\n"]
endfunction

" wrap multiple lines in begin...end block
function! _EscapeText_julia(text)
  if len(split(a:text, "\n")) > 1
    return ["begin\n", a:text."end\n"]
  endif
  return a:text
endfunction