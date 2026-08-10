# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands for Development

### Code Quality and Linting
```bash
# Format code with ruff (primary formatter)
uv run ruff format src

# Check linting issues with ruff (primary linter)
uv run ruff check src

# Fix auto-fixable linting issues
uv run ruff check --fix src

# Run type checking with mypy
uv run mypy src

# Combined quality check
uv run ruff check src && uv run ruff format --check src && uv run mypy src
```

### Testing
```bash
# Run all tests with pytest
uv run pytest src

# Run tests with coverage report
uv run make coverage

# Run single test file
uv run pytest src/wax/stream_test.py

# Run tests with doctests included
uv run pytest --doctest-modules src
```

### Building and Packaging
```bash
# Build package for distribution
uv build

# Install in development mode
uv sync --dev
```

### Documentation
```bash
# Build documentation (full build with notebook execution)
make docs

# Build documentation fast (without executing notebooks)
make docs-fast
```

### Legacy Makefile Commands (being migrated to ruff/uv)
```bash
# Run comprehensive checks (flake8, mypy, tests, format, license)
make act

# Run tests with coverage
make coverage

# Check formatting (uses legacy black/isort, prefer ruff)
make check-format
```

## High-Level Architecture

### Core Design Philosophy
WAX-ML is a **functional programming** library built on JAX with dual backend support (Haiku and Flax) for streaming time-series data processing. It follows a research-oriented design that emphasizes pure functions over object-oriented patterns, with a Flax-based streaming architecture built around a single streaming transform and a library of streaming modules.

### Key Architectural Components

#### 1. Stream Processing (`src/wax/stream.py`)
The heart of WAX-ML's streaming architecture:
- **`Stream` class**: Implements Poincaré-Einstein synchronization for multi-frequency data streams
- **Data tracing mechanism**: Pre-computes indices for efficient JAX-compatible data access
- **Temporal synchronization**: Handles streams with different time resolutions using:
  - Forward-filling for lower frequency data
  - Buffering for higher frequency data
- **Causality preservation**: Ensures no future information leaks into computations

#### 2. Unroll Transformations (`src/wax/unroll.py`)
Generalizes RNN-style sequential processing:
- **`unroll`**: Applies stateful transformations to sequential data
- **`static_scan`**: JAX-optimized scanning for fixed-length sequences
- **Haiku integration**: Works with `transform_with_state` for pure functional state management

#### 3. Data Container Accessors (`src/wax/accessors.py`)
Bridges high-level data APIs with JAX functions:
- **Pandas integration**: `.wax` accessor for DataFrame/Series
- **Xarray integration**: `.wax` accessor for Dataset/DataArray
- **Streaming interface**: `data.wax.stream().apply(function)` pattern
- **Format preservation**: Maintains original data container types in outputs

#### 4. Haiku Modules (`src/wax/modules/`)
Functional building blocks for time-series processing:
- **`EWMA`**: Exponential Moving Average with multiple parameterizations
- **`Buffer`**: Fixed-size buffering for streaming data
- **`UpdateOnEvent`**: Conditional computation updates based on events
- **`OnlineSupervisedLearner`**: Online ML with function optimization
- **Statistical modules**: Rolling statistics, lag operators, differencing

#### 5. Encoding Schemes (`src/wax/encode.py`)
Enables JAX compatibility for non-native types:
- **datetime64 encoding**: Converts to pairs of int32 for JAX compatibility
- **String encoding**: Uses sklearn LabelEncoder for categorical data
- **Reversible transformations**: All encodings support decode operations

#### 6. Flax Streaming Architecture (`src/wax/flax/`)
Streaming computation built on Flax. There is **one** transform, plus a module library:
- **The transform** (`src/wax/flax/core/streaming_transforms.py`):
  `@streaming_transform_with_state` turns a function containing Flax modules into a
  `StreamingTransform` exposing `.init(rng, *args)`, `.apply(params, state, rng, *args)`
  and `.scan(params, state, rng, inputs)`. Everything else in the Flax streaming layer
  is either a module used inside such a function, or a composition built on top of it.
- **Streaming modules** (`src/wax/flax/modules/`): Flax-based equivalents of Haiku modules (EWMA, Buffer, ARMA, etc.)
- **Transform layer** (`src/wax/flax/core/transform.py`): Flax-compatible transform utilities
- **In-tree but not public surface for 0.7.0**: `update_on_event`,
  `ConditionalComputation`, `streaming_scan`, `streaming_optimizer`. None of these is
  exported from `wax.flax.core`; import them from
  `wax.flax.core.streaming_transforms`. See "Symbols that are not peers of the
  transform" below before using any of them.

#### 7. Gym Integration (`src/wax/gym/`)
Reinforcement learning and feedback loop support:
- **`GymFeedback`**: Functional RL environment wrapper
- **Entity system**: Converts functions to stateful Gym-compatible objects
- **Callbacks**: Dask-inspired callback system for monitoring and control

### Data Flow Architecture

#### Typical Streaming Workflow:
1. **Data Ingestion**: xarray.Dataset or pandas containers
2. **Stream Setup**: Configure temporal synchronization with `Stream` class
3. **Function Application**: Apply Haiku modules via `.wax.stream().apply()`
4. **Data Tracing**: Pre-compute access indices for JAX optimization
5. **Sequential Processing**: Use `unroll` for stateful transformations
6. **Output Formatting**: Return results in original container format

#### JAX Integration Patterns:

**Haiku (Legacy)**:
```python
# Transform Haiku module to pure functions
@hk.transform_with_state
def model(x):
    return EWMA(alpha=0.1)(x)

# Apply to streaming data
outputs, final_state = data.wax.stream().apply(model)
```

**Flax (Streaming Architecture)**:
```python
from wax.flax.core import streaming_transform_with_state

# The transform: state management for a function containing Flax modules
@streaming_transform_with_state
def streaming_model(x):
    ewma = EWMA(alpha=0.1)
    buffer = Buffer(maxlen=10)
    return ewma(buffer(x))

# Step-by-step
params, state = streaming_model.init(rng, x0)
output, state = streaming_model.apply(params, state, None, x1)

# Or over a whole sequence, via jax.lax.scan
outputs, final_state = streaming_model.scan(params, state, None, xs)
```

### Dependencies and Ecosystem
- **Core**: JAX, Haiku (legacy), Flax (new streaming architecture)
- **Data**: pandas, xarray (high-level data containers)
- **Optimization**: Optax (gradient-based optimization, integrated with streaming optimizers)
- **ML**: scikit-learn (encoding utilities)
- **Type Safety**: Full mypy coverage with modern type annotations

### Project Structure
- **`src/wax/`**: Main package with functional modules
- **`src/wax/modules/`**: Haiku modules for time-series operations (legacy)
- **`src/wax/flax/`**: **NEW** Flax-based streaming architecture
  - **`src/wax/flax/core/`**: Core streaming transforms and utilities
  - **`src/wax/flax/modules/`**: Flax streaming modules (EWMA, Buffer, ARMA, etc.)
- **`src/wax/gym/`**: Reinforcement learning integration
- **`src/wax/optim/`**: Optimization utilities
- **`src/wax/datasets/`**: Synthetic dataset generators

### Code Quality Standards
- **Modern Python**: Uses `X | Y` union syntax, `collections.abc` imports
- **Type Safety**: Comprehensive mypy coverage with proper stub packages
- **Functional Purity**: All core algorithms are pure functions compatible with JAX transformations
- **Testing**: Extensive test coverage including doctests and property-based testing

The architecture enables efficient processing of streaming time-series data while maintaining functional purity for JAX optimization and hardware acceleration.

## Flax Streaming Architecture

### The transform

There is exactly one transform. `streaming_transform_with_state` wraps a function that
instantiates Flax modules and returns a `StreamingTransform`:

```python
from wax.flax.core import streaming_transform_with_state

@streaming_transform_with_state
def streaming_model(x):
    buffer = Buffer(maxlen=10)
    ewma = EWMA(alpha=0.1)
    return ewma(buffer(x))

params, state = streaming_model.init(rng, x0)          # -> (params, state)
output, state = streaming_model.apply(params, state, None, x1)
outputs, final_state = streaming_model.scan(params, state, None, xs)
```

`.apply` is the single-step form; `.scan` runs the same computation over a leading
sequence axis via `jax.lax.scan`. Both produce identical results.

### The module library (`src/wax/flax/modules/`)

Flax equivalents of the Haiku modules. These are ordinary `nn.Module` classes:
instantiate and call them *inside* a function passed to
`streaming_transform_with_state`.

Re-exported from `wax.flax.modules`: `ARMA`, `ApplyMask`, `Buffer`, `Counter`, `Diff`,
`EWMA`, `EWMCov`, `EWMVar`, `Ffill`, `FillNanInf`, `HasChanged`, `Lag`, `MaskMean`,
`MaskNormalize`, `MaskStd`, `OHLC`, `OnlineOptimizer`, `OptaxOptimizer`, `PctChange`,
`RollingMean`, `VMap`.

Present in the tree but **not** re-exported — import from their own submodule:
`CompressedBuffer` and `HierarchicalBuffer` (`.compressed_buffer`), `SNARIMAX`
(`.snarimax`), `GymFeedback` (`.gym_feedback`), `FuncOptimizer` (`.func_optimizer`),
`UpdateParams` (`.update_params`).

### Symbols that are not peers of the transform

`update_on_event`, `streaming_scan` and `streaming_optimizer` exist in
`src/wax/flax/core/streaming_transforms.py`. They are **not** part of the public
surface for 0.7.0 — `wax.flax.core` exports exactly six names (`FlaxTransformed`,
`flax_transform_with_state`, `flax_unroll`, `flax_unroll_transform`,
`StreamingTransform`, `streaming_transform_with_state`) — and none of them is a fourth
"fundamental transform". Their real shapes:

- **`update_on_event(event_fn=...)`** is decorator sugar over the
  `ConditionalComputation` Flax **module** (also unexported). The decorated object is a plain function
  with no `.init` / `.apply`; calling it directly raises
  `flax.errors.CallCompactUnboundModuleError`. It only works when the decorated
  function is then wrapped by `streaming_transform_with_state`. Note also that the
  current implementation caches the output correctly but does **not** roll back inner
  module state — child modules (EWMA etc.) still advance on non-event steps, unlike
  the Haiku `UpdateOnEvent`.
- **`streaming_scan(reset_on=...)`** returns a non-callable `StreamingScanResult`;
  calling it raises `TypeError: 'StreamingScanResult' object is not callable`. The
  usable entry point is its `scan_apply(inputs, params=None, state=None, rng=None)`
  method. Conceptually this is the `.scan` method of `StreamingTransform` plus a reset
  predicate, not a separate concept. Its reset is also known-wrong: it rewinds to a
  state that has already consumed `inputs[0]`, not to a fresh module (xfail test in
  `streaming_scan_test.py`).
- **`streaming_optimizer(optimizer, loss_fn)`** is a composite: the transform applied
  over the `StreamingOptimizer` module. It is *not* exported from `wax.flax.core` —
  `from wax.flax.core import streaming_optimizer` raises `ImportError`; import it from
  `wax.flax.core.streaming_transforms`. It differentiates a single scalar scale
  parameter, not the model's own parameters — and returns the *unscaled* output, so
  even that scalar never reaches the caller's prediction.

See `ROADMAP.md` Phase 1.2 (`update_on_event` state rollback) and Phase 1.3
(`streaming_optimizer` full-parameter gradients) for the work that would make these
behave as their names suggest.

### Examples and demonstrations

- `src/wax/flax/examples/transform_compositions.py` — technical indicators, trading systems
- `src/wax/flax/examples/jax_scan_compatibility.py` — performance optimization patterns
- The test suites next to each module, which are the most reliable usage reference

### Key properties

- **JAX Compatibility**: `StreamingTransform` works under JIT and `jax.lax.scan`
- **State Management**: Automatic state threading across time steps
- **Type Safety**: mypy coverage with modern type annotations

### Testing Flax Modules

When working with Flax streaming modules, use these test patterns:
```bash
# Run all Flax streaming tests
uv run pytest src/wax/flax/ -v

# Test specific streaming transforms
uv run pytest src/wax/flax/core/streaming_transforms_test.py -v

# Test streaming optimizer
uv run pytest src/wax/flax/core/streaming_optimizer_test.py -v

# Test Flax modules (EWMA, Buffer, etc.)
uv run pytest src/wax/flax/modules/ -v

# Test advanced state patterns
uv run pytest src/wax/flax/core/advanced_state_patterns_test.py -v

# Test debugging and profiling tools
uv run pytest src/wax/flax/debug/debug_test.py -v

# Test visualization tools
uv run pytest src/wax/flax/visualization/visualization_test.py -v
```

## Advanced Streaming Features

These are experimental in-tree surfaces, not part of the 0.7.0 public API. Check the
adjacent test file before relying on any of them.

### Advanced State Patterns (`wax.flax.core.advanced_state_patterns`)

State management patterns for complex streaming systems:

#### Hierarchical State Machines
Multi-level state coordination with dependency management:
```python
from wax.flax.core.advanced_state_patterns import streaming_state_machine

@streaming_state_machine({
    'market': MarketRegimeDetector(),
    'volatility': VolatilityRegimeDetector()
}, dependencies={'volatility': ['market']})
def multi_regime_trading_system(state_outputs, price, volume):
    # State machine handles coordination automatically
    return combine_regime_signals(state_outputs, price, volume)
```

#### Attention-Based State Selection
Dynamic attention to relevant historical states:
```python
from wax.flax.core.advanced_state_patterns import streaming_attention_state

@streaming_attention_state(embed_dim=64, max_history=100)
def adaptive_context_processor(attention_output, signal):
    # Function receives enhanced state with attention context
    enhanced_state = attention_output["enhanced_state"]
    return process_with_context(enhanced_state, signal)
```

#### Compositional State Patterns
Building complex systems from simple components:
```python
from wax.flax.core.advanced_state_patterns import streaming_compose_states

@streaming_compose_states(
    TrendAnalyzer(),
    MomentumAnalyzer(),
    strategy="pipeline"
)
def integrated_analysis_system(composed_output, price, volume):
    # All state patterns are composed automatically
    return combine_analysis_results(composed_output, price, volume)
```

### Development Tools

Comprehensive debugging and profiling tools for streaming computation:

#### Streaming Debugger
Real-time state inspection with conditional breakpoints:
```python
from wax.flax.debug import StreamingDebugger, debug_streaming

debugger = StreamingDebugger()
debugger.add_breakpoint("high_value", lambda step, state, inp, out: inp > 100)

@debug_streaming(debugger, "my_module")
@streaming_transform_with_state
def my_streaming_fn(x):
    return EWMA(alpha=0.1)(x)
```

#### Performance Profiler
Execution time analysis and bottleneck identification:
```python
from wax.flax.debug import StreamingProfiler, profile_streaming

profiler = StreamingProfiler()

@profile_streaming(profiler, "ewma_module")
@streaming_transform_with_state  
def ewma_processor(x):
    return EWMA(alpha=0.1)(x)

# Get performance report
results = profiler.finalize()
print(results.get_summary())
```

#### Memory Tracker
Memory usage analysis and leak detection:
```python
from wax.flax.debug import MemoryTracker, track_memory_usage

tracker = MemoryTracker(enable_detailed_tracking=True)

@track_memory_usage(tracker, "buffer_module")
@streaming_transform_with_state
def buffer_processor(x):
    return Buffer(maxlen=10)(x)

print(tracker.generate_memory_report())
```

### Pipeline Visualization

Comprehensive visualization tools for monitoring and analyzing streaming pipelines:

#### Computation Graph Rendering
Visualize pipeline structure and dependencies:
```python
from wax.flax.visualization import render_pipeline_graph

# Render pipeline graph to various formats
render_pipeline_graph(
    streaming_function, 
    input_example,
    output_path="pipeline.png",
    format="png",
    include_shapes=True
)
```

#### Real-time Data Flow Visualization
Monitor streaming data through pipeline components:
```python
from wax.flax.visualization import DataFlowTracker, visualize_streaming_data

# Track data flow
tracker = DataFlowTracker()
tracker.record_data("module1", "input", data_value)

# Visualize streaming data
plot = visualize_streaming_data(tracker, backend="matplotlib")
```

#### Interactive Web Dashboard
Production-ready monitoring interface:
```python
from wax.flax.visualization import InteractiveDashboard, DashboardConfig

# Create dashboard
config = DashboardConfig(host="0.0.0.0", port=8080)
dashboard = InteractiveDashboard(config)

# Register pipelines
dashboard.register_pipeline("trading_system", trading_fn, input_example)

# Start monitoring server
dashboard.start_server()
# Visit http://localhost:8080 for real-time monitoring
```

### Modern Jupyter Visualizations

State-of-the-art interactive visualization tools designed specifically for Jupyter notebooks:

#### Interactive Plotly Visualizations
Real-time web-based charts with full interactivity:
```python
from wax.flax.visualization import quick_pipeline_viz, quick_streaming_plot

# One-line pipeline visualization
quick_pipeline_viz(streaming_function, input_example)

# Real-time streaming plot
fig, viz = quick_streaming_plot("Pipeline Data")
viz.add_stream(fig, "Signal", "#1f77b4")
viz.update_stream(fig, "Signal", value, timestamp)
```

#### Interactive Parameter Controls
Real-time parameter tuning with ipywidgets:
```python
from wax.flax.visualization import InteractiveParameterControls

controls = InteractiveParameterControls()
params = {
    'alpha': {'type': 'float', 'min': 0.01, 'max': 0.5, 'value': 0.1},
    'window_size': {'type': 'int', 'min': 5, 'max': 100, 'value': 20}
}
panel = controls.create_parameter_panel(params)

# Add callback for parameter changes
controls.add_callback(lambda params: update_pipeline(params))
```

#### High-Performance Bokeh Streaming
Ultra-fast WebGL-accelerated visualizations:
```python
from wax.flax.visualization import BokehStreamingPlot, create_bokeh_streaming_demo

# High-performance streaming plot
plot = BokehStreamingPlot()
bokeh_fig = plot.create_streaming_line_plot("Real-time Data")
plot.add_line_stream(bokeh_fig, "Fast Stream", "#ff7f0e")

# Complete streaming demo
demo_layout = create_bokeh_streaming_demo(data_tracker)
```

#### Animated Pipeline Flow
Beautiful animated data flow visualizations:
```python
from wax.flax.visualization import AnimatedPipelineFlow

animator = AnimatedPipelineFlow()
stages = ["Input", "Processing", "Analysis", "Output"]
flow_fig = animator.create_flow_animation(stages, data_tracker)
animator.start_animation(data_tracker)
```

#### 3D State Space Exploration
Interactive 3D visualizations of model states:
```python
# Available in Jupyter with Plotly
# Creates interactive 3D scatter plots with:
# - State evolution trajectories
# - Color-coded by additional dimensions
# - Interactive rotation and zoom
# - Hover tooltips with detailed information
```

#### Comprehensive Interactive Dashboard
Complete monitoring solution with all components:
```python
from wax.flax.visualization import create_jupyter_dashboard, display_pipeline_dashboard

# Create complete dashboard
dashboard = create_jupyter_dashboard(pipeline_function, input_example)

# Display in Jupyter
display_pipeline_dashboard(dashboard)
# Includes: pipeline graph, streaming plots, parameter controls, performance metrics
```

### Installation Requirements for Modern Visualizations

```bash
# Core modern visualization features
pip install plotly ipywidgets networkx

# High-performance streaming (optional)
pip install bokeh

# For Jupyter notebook support  
pip install jupyter jupyterlab
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Enable widget extensions
jupyter nbextension enable --py widgetsnbextension
```

### Demo Notebooks

Comprehensive demonstration materials are available:

#### Advanced State Patterns Demo
- **Location**: `notebooks/advanced_state_patterns_demo.py`
- **Features**: Hierarchical state machines, attention mechanisms, compositional patterns
- **Use Cases**: Financial market analysis, multi-regime systems, integrated analysis

#### Debugging and Profiling Demo  
- **Location**: `notebooks/debugging_and_profiling_demo.py`
- **Features**: Real-time debugging, performance profiling, memory tracking
- **Use Cases**: Production monitoring, performance optimization, bottleneck analysis

#### Pipeline Visualization Demo
- **Location**: `notebooks/pipeline_visualization_demo.py`
- **Features**: Graph rendering, data flow visualization, interactive dashboards
- **Use Cases**: System monitoring, pipeline analysis, production deployment

#### Modern Interactive Visualization Demo
- **Location**: `notebooks/modern_visualization_demo.py`
- **Features**: Plotly interactive charts, Bokeh streaming, ipywidgets controls, 3D visualizations
- **Use Cases**: Real-time monitoring, parameter tuning, advanced analysis, presentation-quality outputs

### Testing Advanced Features

```bash
# Test all advanced features
uv run pytest src/wax/flax/core/advanced_state_patterns_test.py src/wax/flax/debug/debug_test.py src/wax/flax/visualization/visualization_test.py -v

# Run demo notebooks (requires Jupyter or run as Python scripts)
uv run python notebooks/advanced_state_patterns_demo.py
uv run python notebooks/debugging_and_profiling_demo.py  
uv run python notebooks/pipeline_visualization_demo.py
uv run python notebooks/modern_visualization_demo.py

```