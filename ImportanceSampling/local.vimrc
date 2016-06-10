augroup switchtmux
	au!
	autocmd BufEnter *.R   silent !tmux select-window -t iSampling:R
	autocmd BufEnter *.jl  silent !tmux select-window -t iSampling:julia
	autocmd BufEnter *.py  silent !tmux select-window -t iSampling:ipython
	autocmd BufEnter *.sql silent !tmux select-window -t iSampling:sqlite
	autocmd BufEnter *.R   let b:slime_config = {"socket_name": "default", "target_pane": "iSampling:R"}
	autocmd BufEnter *.jl  let b:slime_config = {"socket_name": "default", "target_pane": "iSampling:julia"}
	autocmd BufEnter *.py  let b:slime_config = {"socket_name": "default", "target_pane": "iSampling:ipython"}
	autocmd BufEnter *.sql let b:slime_config = {"socket_name": "default", "target_pane": "iSampling:sqlite"}
augroup END

augroup juliamodule
	au!
	autocmd BufEnter supply-demand.jl nnoremap <buffer> <leader>n :w<bar>SlimeSend1 include("supply-demand.jl")<CR>
augroup END
	
