# Development Guide

This guide provides instructions for code quality tools and development workflows for WAX-ML.

## Code Quality Tools

### Ruff (Linting and Formatting)

Ruff is configured as the primary linter and formatter for this project.

#### Check for linting issues:
```bash
uv run ruff check src
```

#### Fix auto-fixable linting issues:
```bash
uv run ruff check --fix src
```

#### Format code:
```bash
uv run ruff format src
```

#### Check both linting and formatting:
```bash
uv run ruff check src && uv run ruff format --check src
```

### MyPy (Type Checking)

MyPy is configured for type checking with proper stub packages installed.

#### Run type checking:
```bash
uv run mypy src
```

#### Run type checking with verbose output:
```bash
uv run mypy src --verbose
```

#### Run type checking on specific files:
```bash
uv run mypy src/wax/stream.py
```

## Configuration

### Ruff Configuration
Ruff is configured in `pyproject.toml`:
- Line length: 100 characters
- Target Python version: 3.10+
- Selected rules: E, W, F, I, B, C4, UP
- Automatic import sorting and code modernization

### MyPy Configuration
MyPy is configured in `pyproject.toml`:
- Python version: 3.10
- Graceful handling of missing imports for external libraries
- Type stubs installed for: pandas, tqdm, scipy, openpyxl, python-dateutil

## Installed Type Stubs

The following type stub packages are installed for better type checking:
- `pandas-stubs` - Type stubs for pandas
- `types-tqdm` - Type stubs for tqdm
- `scipy-stubs` - Type stubs for scipy
- `types-openpyxl` - Type stubs for openpyxl
- `types-python-dateutil` - Type stubs for python-dateutil

## Development Workflow

### Before Committing Code

1. **Format code with ruff:**
   ```bash
   uv run ruff format src
   ```

2. **Check and fix linting issues:**
   ```bash
   uv run ruff check --fix src
   ```

3. **Run type checking:**
   ```bash
   uv run mypy src
   ```

4. **Run tests:**
   ```bash
   uv run pytest
   ```

### Quick Quality Check

Run all quality checks at once:
```bash
uv run ruff check src && uv run ruff format --check src && uv run mypy src
```

## Current Status

✅ **Ruff:** All linting rules pass  
✅ **MyPy:** All type checking passes (91 source files checked)  
✅ **Code Modernization:** Complete migration to modern Python syntax  

## Notes

- The project uses modern Python type annotations (e.g., `X | Y` instead of `Union[X, Y]`)
- Some external libraries (optax, sklearn, etc.) don't have type stubs, so they're configured to ignore missing imports
- Test files have relaxed linting rules where appropriate
- All development dependencies are managed through `uv` and defined in `pyproject.toml`