TARGETS = $(wildcard ./*.py)
TARGETS_IPYNB = $(TARGETS:.py=.ipynb)
IPYKERNEL_NAME = python3
PACKAGE_NAME = './'

.PHONY: all
all: run

.PHONY: sync
sync:
	jupytext --sync -- $(TARGETS)

.PHONY: formats
formats: 
	for f in $(TARGETS_IPYNB); do jupytext --set-formats ipynb,py,md:myst $${f}; done

.PHONY: run
run: sync
	papermill 01_*.ipynb /dev/null 
	papermill 02_*.ipynb /dev/null 
	papermill 03_*.ipynb /dev/null 
	papermill 04_*.ipynb /dev/null -p T 100 -p N 10
	$(MAKE) sync

.PHONY: kernel
kernel:
	jupytext --set-kernel $(IPYKERNEL_NAME) -- $(TARGETS)
	$(MAKE) sync

.PHONY: autoflake
autoflake: 
	autoflake --in-place --remove-all-unused-imports -r $(PACKAGE_NAME)
	$(MAKE) sync

.PHONY: isort
isort:
	python -m isort --profile black --treat-comment-as-code "# %%" $(PACKAGE_NAME)
	$(MAKE) sync

.PHONY: black
black:
	python -m black $(PACKAGE_NAME)
	$(MAKE) sync

.PHONY: format
format: autoflake isort black
