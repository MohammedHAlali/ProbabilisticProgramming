
let b:auBufEnter='autocmd BufEnter ' . expand('<sfile>:p:h')
augroup switchtmux-variational
	au!
	exe b:auBufEnter . '/*.jl  silent !tmux select-window -t variational:julia'
	exe b:auBufEnter . '/*.jl  let b:slime_config = {"socket_name": "default", "target_pane": "variational:julia.0"}'
augroup END

" RCall R-block in julia REPL
function! _EscapeText_r(text)
  return ["R\"\"\"\n", a:text."\"\"\"\n"]
endfunction

function! EvalInREPL()
  SlimeSend1 include("variational.jl")
  edit repl.jl
  normal 'z
  execute "normal \<Plug>SlimeParagraphSend"
  sleep 500m
  silent make capture
  edit variational.jl
endfunction

augroup EvalInREPL
  au!
  " autocmd BufWrite variational.jl :call EvalInREPL()
augroup END

