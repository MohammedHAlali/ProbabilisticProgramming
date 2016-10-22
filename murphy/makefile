all: capture

mem: memory-profile.jl
	julia --track-allocation=user $<

profile: /tmp/timing-profile.txt
	awk '{ print $$4" "$$2" "$$1" "$$3; }' $<
	# cat $

test:
	julia runtests.jl

capture:
	@tmux capture-pane -p -S -20 -J -t 0 | \
	awk -v RS='julia> ' '\
		BEGIN { ORS=RS } \
			{b=a; a=$$0} \
		END { print b } '

clean:
	rm -v *.jl.mem

lint:

	julia -E 'using Lint; lintfile("$<")'

tags:
	ctags-exuberant -R --totals=yes --languages=julia

.PHONY: mem profile test capture clean lint tags
