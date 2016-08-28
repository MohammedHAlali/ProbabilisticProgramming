augroup switchtmux
	au!
	autocmd BufEnter *.R   silent !tmux select-window -t variational:R
	autocmd BufEnter *.jl  silent !tmux select-window -t variational:julia
	autocmd BufEnter *.R   let b:slime_config = {"socket_name": "default", "target_pane": "variational:R"}
	autocmd BufEnter *.jl  let b:slime_config = {"socket_name": "default", "target_pane": "variational:julia"}
augroup END

