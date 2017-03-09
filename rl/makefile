all: capture

test:
	python3 bandit-test.py
	# python3 -m unittest discover -p '*-test.py'

capture:
	@tmux capture-pane -p -S -20 -J -t 0 | \
	awk -v RS='In ' '\
		BEGIN { ORS=RS } \
			{b=a; a=$$0} \
		END { print b } '
