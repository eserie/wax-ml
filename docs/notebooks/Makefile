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
	mkdir -p _tmp
	papermill 01_*.ipynb _tmp/run_01.ipynb
	papermill 02_*.ipynb _tmp/run_02.ipynb
	papermill 03_*.ipynb _tmp/run_03.ipynb
	papermill 04_*.ipynb _tmp/run_04.ipynb -p T 100 -p N 10
	papermill 05_*.ipynb _tmp/run_05.ipynb -p TRAIN_STEPS 512 -p TOTAL_LEN 1000 -p TRAIN_SIZE 512 -p BATCH_SIZE 2 -p SEQ_LEN 32 -p TRAIN_DATE "2009-03-07"
	papermill 06_*.ipynb _tmp/run_06.ipynb
	papermill 07-*.ipynb _tmp/run_07.ipynb -p N_STEP_SIZE 2 -p N_EPS 2 -p N_BATCH 2  -p T 100
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
