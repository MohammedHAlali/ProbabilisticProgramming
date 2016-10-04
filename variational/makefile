all: capture

mem: memory-profile.jl
	julia --track-allocation=user \
		-e 'include("variational.jl"); MixtureModels.profile_negloglikelihood()'

profile: /tmp/prof.txt
	awk '{ print $$3" "$$2" "$$1" "$$4; }' $<
	# cat $<

test:
	julia runtests.jl

capture:
	@tmux capture-pane -p -S -20 -J -t 0 | \
	awk -v RS='julia> ' '\
		BEGIN { ORS=RS } \
			{b=a; a=$$0} \
		END { print b } '

capture-error:
	tmux capture-pane -p | awk '\
		/ERROR:/ { print $$0; next} \
		/^ in/ && x=="" { x=$$0; next } \
		/^ in/ && x!="" { print x; x=$$0; next } \
		/^$$/ { print x; x=""; next } \
		x!="" {x=x $$0} '

clean:
	rm *.jl.mem

lint: runtests.jl
	julia -E 'using Lint; lintfile("$<")'

tags:
	ctags-exuberant -R --totals=yes --languages=julia

.PHONY: tags
