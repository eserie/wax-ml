# Demo notebooks

These demos are versioned as [jupytext](https://jupytext.readthedocs.io) `.py`
files in the `percent` format, not as `.ipynb`. Keeping the source in plain
Python makes diffs readable and keeps execution outputs (which embed local
paths and large base64 blobs) out of the repository.

To work with them in Jupyter, generate the notebooks first:

```bash
make notebooks        # jupytext --to ipynb notebooks/*.py
jupyter lab notebooks/
```

The generated `.ipynb` files are gitignored. Edits made in Jupyter are written
back to the paired `.py` file by jupytext, so commit the `.py` only.

To remove the generated notebooks:

```bash
make clean-notebooks
```

The tutorial notebooks under `docs/notebooks/` follow a different convention:
they keep their `.ipynb` because the Sphinx/myst-nb documentation build renders
them with their stored outputs.
