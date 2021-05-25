# You can set these variables from the command line.
DOCS_DIR = docs
SPHINXBUILD = sphinx-build
GRAPHVIZ_DOT = dot
PACKAGE_NAME = wax
IPYKERNEL_NAME = python3

# Always rebuild these targets
.PHONY: all

.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo
	@echo "	 act		 Run actions: flake8, mypy, tests"
	@echo "	 flake8		 Run the flake8."
	@echo "	 format		 Format python files with autofloke, isort and black."
	@echo "	 tests		 Run the tests."
	@echo "	 coverage	Generate a coverage report for unit tests."
	@echo "	 docs		Build the documentation in the docs/_build/html/ directory."
	@echo "	 clean		Remove documentation build, coverage and Python stuff."
	@echo "	 release	Create a source distribution in the dist/ directory."
	@echo

.PHONY: autoflake
autoflake:
	autoflake --in-place --remove-all-unused-imports -r $(PACKAGE_NAME)

.PHONY: isort
isort:
	python -m isort --profile black --float-to-top $(PACKAGE_NAME)

.PHONY: black
black:
	python -m black $(PACKAGE_NAME)

.PHONY: format-package
format-package: autoflake isort black

.PHONY: format-notebooks
format-notebooks :
	cd docs/notebooks && $(MAKE) format

.PHONY: format
format: format-package format-notebooks

.PHONY: check-format
check-format: format
	git diff
	./build/has_changed.sh

.PHONY: flake8
flake8:
	# stop the build if there are Python syntax errors or undefined names
	python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# all Python files should follow PEP8 (except some notebooks, see setup.cfg)
	python -m flake8 $(PACKAGE_NAME)
	# exit-zero treats all errors as warnings.  The GitHub editor is 127 chars wide
	# flake8 . --count --exit-zero --max-complexity=10 --statistics


.PHONY: mypy
mypy:
	python -m mypy -p $(PACKAGE_NAME)


.PHONY: tests
tests:
	python -m py.test \
	--doctest-modules \
	-n auto \
	--durations=30 \
	$(PACKAGE_NAME)

.PHONY: coverage
coverage:
	python -m py.test \
	--cov ./ \
	--cov-report xml  \
	--cov-report html  \
	--cov-report term \
	--cov $(PACKAGE_NAME) \
	--doctest-modules \
	-n auto \
	$(PACKAGE_NAME)

.PHONY: docs
docs:
	cd docs && $(MAKE) html

.PHONY: docs-fast
docs-fast:
	sphinx-build -b html -D jupyter_execute_notebooks=off docs docs/_build/html

.PHONY: env
release:
	virualenv venv

.PHONY: release
release:
	python setup.py sdist

.PHONY: clean-py
clean-py:
	find . \( -name '*.pyc' -o -name '*.py,cover' -o -name '*~' -o -name '*__pycache__' \) -prune -exec rm -rf {} \;
	rm -rf *.egg-info dist

.PHONY: clean-doc
clean-doc:
	rm -rf docs/_build/*

.PHONY: clean-coverage
clean-coverage:
	rm -rf htmlcov/

.PHONY: clean
clean: clean-docs clean-py clean-coverage

.PHONY: distclean
distclean: clean clean-venv

.PHONY: tags
tags: clean
	ctags -e -R --languages=python

.PHONY: ipykernel
ipykernel:
	python -m ipykernel install --user --name $(IPYKERNEL_NAME)

.PHONY: license
license:
	./build/set_license.sh

.PHONY: check-license
check-license: license
	./build/has_changed.sh

.PHONY: run-notebooks
run-notebooks :
	cd docs/notebooks && $(MAKE) run

.PHONY: act
act : flake8 mypy check-license check-format coverage run-notebooks

.PHONY: pre-commit
pre-commit:
	pre-commit run --all
