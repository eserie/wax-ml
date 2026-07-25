# Installing `wax`

First, obtain the WAX-ML source code:

```
git clone https://github.com/eserie/wax-ml
cd wax
```

You can install `wax` by running:
```bash
pip install -e .[complete]  # install wax
```

To upgrade to the latest version from GitHub, just run `git pull` from the WAX-ML
repository root. You shouldn't have to reinstall `wax` because `pip install -e`
sets up symbolic links from site-packages into the repository.

You can install `wax` development tools by running:
```bash
pip install -e .[dev]  # install wax-development-tools
```

# Running the tests

To run all the WAX-ML tests, we recommend using `pytest-xdist`, which can run tests in
parallel. First, install `pytest-xdist` and `pytest-benchmark` by running
`ip install -r build/test-requirements.txt`.
Then, from the repository root directory run:

```
pytest -n auto .
```

You can run a more specific set of tests using
[pytest](https://docs.pytest.org/en/latest/usage.html#specifying-tests-selecting-tests)'s
built-in selection mechanisms, or alternatively you can run a specific test
file directly to see more detailed information about the cases being run:

```bash
pytest -v wax/accessors_test.py
```

The Colab notebooks are tested for errors as part of the documentation build and Github actions.


# Type checking

We use `mypy` to check the type hints. To check types locally the same way
as Github actions checks, you can run:
```
mypy wax
```
or
```
make mypy
```

# Flake8

We use `flake8` to check that the code follow the pep8 standard.
To check the code, you can run
```
make flake8
```

# Formatting code

We use `isort` and `black` to format the code.

When you are in the root directory of the project,
to format code in the package, you can run:

```bash
make format-package
```

To format notebooks in the documentation, you can use:
```bash
make format-notebooks
```

To format all files you can run:
```bash
make format
```

Note that the CI running with actions will verify that formatting
all source code does not affect the files.
You can check this locally by running :
```bash
make check-format
```

# Check actions
You can check that everything is ok by running:
```bash
make act
```
This will check flake8, mypy, isort and black formatting, licenses headers
and run tests and coverage.

# Update documentation

To rebuild the documentation, install several packages:
```
pip install -r docs/requirements.txt
```
And then run:
```
sphinx-build -b html docs docs/build/html
```
or run
```bash
make docs
```

The first such build takes a long time, because it executes the notebooks in the
documentation source. Subsequent builds replay them from the execution cache; see
[Notebooks within the sphinx build](#notebooks-within-the-sphinx-build) below.

If you'd prefer to build the docs without executing the notebooks at all -- much
faster, but the tutorials render without their outputs -- you can run:
```
sphinx-build -b html -D nb_execution_mode=off docs docs/_build/html
```
or run
```bash
make docs-fast
```
You can then see the generated documentation in `docs/_build/html/index.html`.

## Update notebooks

We use [jupytext](https://jupytext.readthedocs.io/) to maintain three synced copies of the notebooks
in `docs/notebooks`: one in `ipynb` format, one in `py` and one in `md` format.
The advantage of the former is that it can be opened and executed directly in Colab;
the advantage of the second is that it makes easier to refactor and format python code;
the advantage of the latter is that it makes it much easier to track diffs within version control.

### Editing ipynb

For making large changes that substantially modify code and outputs, it is easiest to
edit the notebooks in Jupyter or in Colab. To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb`.
You may want to test that it executes properly, using `sphinx-build` as explained above.

You could format the python code in your notebooks by running `make format`
in the `docs/notebooks` directory
or
`make format-notebooks`
in the root directory.


### Editing md

For making smaller changes to the text content of the notebooks, it is easiest to edit the
`.md` versions using a text editor.

### Syncing notebooks

After editing either the ipynb or md versions of the notebooks, you can sync the two versions
using [jupytext](https://jupytext.readthedocs.io/) by running:

```bash
jupytext --sync docs/notebooks/*
```
or:
```bash
cd  docs/notebooks/
make sync
```

Alternatively, you can run this command via the [pre-commit](https://pre-commit.com/)
framework by executing the following in the main WAX-ML directory:

```bash
pre-commit run --all
```

See the pre-commit framework documentation for information on how to set your local git
environment to execute this automatically.

### Creating new notebooks

If you are adding a new notebook to the documentation and would like to use the `jupytext --sync`
command discussed here, you can set up your notebook for jupytext by using the following command:

```bash
jupytext --set-formats ipynb,py,md:myst path/to/the/notebook.ipynb
```

This works by adding a `"jupytext"` metadata field to the notebook file which specifies the
desired formats, and which the `jupytext --sync` command recognizes when invoked.

### Notebook outputs are not versioned

Notebook outputs are build artifacts, not sources. They are stripped from the
repository by the [nbstripout](https://github.com/kynan/nbstripout) pre-commit hook
and regenerated by the documentation build.

Two reasons. Stored outputs make diffs unreadable -- a re-run rewrites base64 image
blobs and execution counts even when no code changed. And they leak: a warning
traceback embeds the absolute path of whoever ran the notebook, which then ships
with the published documentation.

If an output ever slips through, this fails:

```bash
make check-notebooks
```

and this fixes it:

```bash
make strip-notebooks
```

### Notebooks within the sphinx build

The tutorials under `docs/notebooks` are executed during the documentation build.
The build fails if a cell raises. If an error is intentional, either catch it or tag
the cell with `raises-exceptions` metadata
([example PR](https://github.com/google/jax/pull/2402/files)). You have to add this
metadata by hand in the `.ipynb` file; it is preserved when somebody re-saves the
notebook.

Execution is governed by `nb_execution_mode = "cache"` in
[conf.py](https://github.com/eserie/wax-ml/blob/main/docs/conf.py).
[jupyter-cache](https://jupyter-cache.readthedocs.io/) keys its entries on notebook
content, so a notebook is re-executed only when its code changes and replayed from
the cache otherwise. A build that touches only prose costs about as much as the prose
it renders. The cache lives in `.jupyter_cache/` at the repository root -- outside the
Sphinx source directory, so Sphinx does not mistake the executed notebooks it stores
for source documents, and outside `docs/_build`, so `make clean` cannot discard it.

Note that these option names carry the `nb_` prefix introduced in MyST-NB v0.14.
The older unprefixed spellings are silently ignored rather than rejected, so a
configuration written against them looks correct while doing nothing at all.

## Documentation publishing

The documentation is built and published by the
[docs workflow](https://github.com/eserie/wax-ml/blob/main/.github/workflows/docs.yml)
on every push to `main`, and served from GitHub Pages at
<https://eserie.github.io/wax-ml/>.

A pull request builds the documentation but does not publish it, so a broken build is
caught before it can reach the site.

Publishing happens from CI rather than from a hosted documentation builder because the
tutorials are executed at full size. That cost is only bearable with a persistent
execution cache, and the workflow keeps one across runs -- along with the dataset that
notebook 05 downloads, so a build does not depend on a third-party host being
reachable. The first run after a notebook changes pays the full execution cost; the
ones after it do not.
