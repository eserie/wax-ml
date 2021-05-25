# Installing `wax`

First, obtain the WAX source code:

```
git clone https://github.com/eserie/wax-ml
cd wax
```

You can install `wax` by running:
```bash
pip install -e .[complete]  # install wax
```

To upgrade to the latest version from GitHub, just run `git pull` from the WAX
repository root. You shouldn't have to reinstall `wax` because `pip install -e`
sets up symbolic links from site-packages into the repository.

You can install `wax` development tools by running:
```bash
pip install -e .[dev]  # install wax-development-tools
```

# Running the tests

To run all the WAX tests, we recommend using `pytest-xdist`, which can run tests in
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

This can take a long time because it executes many of the notebooks in the documentation source;
if you'd prefer to build the docs without executing the notebooks, you can run:
```
sphinx-build -b html -D jupyter_execute_notebooks=off docs docs/build/html
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
framework by executing the following in the main WAX directory:

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

### Notebooks within the sphinx build

Some of the notebooks are built automatically as part of the Travis pre-submit checks and
as part of the [Read the docs](https://wax.readthedocs.io/en/latest) build.
The build will fail if cells raise errors. If the errors are intentional,
you can either catch them, or tag the cell with `raises-exceptions` metadata ([example PR](https://github.com/google/jax/pull/2402/files)).
You have to add this metadata by hand in the `.ipynb` file. It will be preserved when somebody else
re-saves the notebook.

We exclude some notebooks from the build, e.g., because they contain long computations.
See `exclude_patterns` in [conf.py](https://github.com/eserie/wax-ml/blob/master/docs/conf.py).

## Documentation building on readthedocs.io

WAX's auto-generated documentations is at <https://wax-ml.readthedocs.io/>.

The documentation building is controlled for the entire project by the
[readthedocs WAX settings](https://readthedocs.org/dashboard/wax-ml). The current settings
trigger a documentation build as soon as code is pushed to the GitHub `main` branch.
For each code version, the building process is driven by the
`.readthedocs.yml` and the `docs/conf.py` configuration files.

For each automated documentation build you can see the
[documentation build logs](https://readthedocs.org/projects/wax-ml/builds/).

If you want to test the documentation generation on Readthedocs, you can push code to the `test-docs`
branch. That branch is also built automatically, and you can
see the generated documentation [here](https://wax-ml.readthedocs.io/en/test-docs/). If the documentation build
fails you may want to [wipe the build environment for test-docs](https://docs.readthedocs.io/en/stable/guides/wipe-environment.html).

For a local test, you can do it in a fresh directory by replaying the commands
executed by Readthedocs and written in their logs:
```
mkvirtualenv wax-docs  # A new virtualenv
mkdir wax-docs  # A new directory
cd wax-docs
git clone --no-single-branch --depth 50 https://github.com/eserie/wax-ml
cd wax
git checkout --force origin/test-docs
git clean -d -f -f
workon wax-docs

python -m pip install --upgrade --no-cache-dir pip
python -m pip install --upgrade --no-cache-dir -I Pygments==2.3.1 setuptools==41.0.1 docutils==0.14 mock==1.0.1 pillow==5.4.1 alabaster>=0.7,<0.8,!=0.7.5 commonmark==0.8.1 recommonmark==0.5.0 'sphinx<2' 'sphinx-rtd-theme<0.5' 'readthedocs-sphinx-ext<1.1'
python -m pip install --exists-action=w --no-cache-dir -r docs/requirements.txt
cd docs
python `which sphinx-build` -T -E -b html -d _build/doctrees-readthedocs -D language=en . _build/html
```


