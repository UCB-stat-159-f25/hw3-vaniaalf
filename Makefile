.PHONY: env
env:
	conda env update --file environment.yml --prune

.PHONY: html
html:
	myst build --html

.PHONY: clean
clean:
	rm -rf figures audio _build