.PHONY: install test demo plots build clean

install: ## install in editable mode with the console script
	pip install -e .

test: ## run the unit-test suite
	python3 -m unittest discover -s tests -v

demo: ## run the full compression pipeline
	modelshrink demo

plots: ## run the pipeline and save charts to docs/assets
	modelshrink demo --plots docs/assets

build: ## build sdist and wheel
	python3 -m build

clean: ## remove build/cache artifacts
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
