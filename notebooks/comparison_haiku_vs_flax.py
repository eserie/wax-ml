# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.004387, "end_time": "2025-07-01T22:54:37.845480", "exception": false, "start_time": "2025-07-01T22:54:37.841093", "status": "completed"}
# # Haiku vs Flax Implementation Comparison
#
# This notebook demonstrates the architectural differences between the original 
# Haiku-based WAX-ML implementation and the new Flax-based parallel implementation.
#
# We'll compare:
# 1. **State Management**: Implicit (Haiku) vs Explicit (Flax)
# 2. **Transform System**: `hk.transform_with_state` vs `flax_transform_with_state`
# 3. **Module Definition**: `hk.Module` vs `flax.linen.Module`
# 4. **Sequential Processing**: Both using the same unroll patterns
# 5. **Performance and Memory Usage**

# %% papermill={"duration": 0.334753, "end_time": "2025-07-01T22:54:38.183236", "exception": false, "start_time": "2025-07-01T22:54:37.848483", "status": "completed"}
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

print(f"JAX backend: {jax.default_backend()}")

# %% papermill={"duration": 0.219636, "end_time": "2025-07-01T22:54:38.404285", "exception": false, "start_time": "2025-07-01T22:54:38.184649", "status": "completed"}
# Generate sample time series data
rng = jax.random.PRNGKey(42)
n_timesteps = 1000

# Create sample data with some NaN values
data = jax.random.normal(rng, (n_timesteps,)) * 0.1 + 0.05
# Add some NaN values to test ignore_na functionality
nan_indices = jax.random.choice(rng, n_timesteps, (50,), replace=False)
data = data.at[nan_indices].set(jnp.nan)

print(f"Data shape: {data.shape}")
print(f"NaN count: {jnp.sum(jnp.isnan(data))}")

# %% [markdown] papermill={"duration": 0.001235, "end_time": "2025-07-01T22:54:38.407012", "exception": false, "start_time": "2025-07-01T22:54:38.405777", "status": "completed"}
# ## 1. Haiku Implementation
#
# Traditional WAX-ML using Haiku's implicit state management:

# %% papermill={"duration": 0.673014, "end_time": "2025-07-01T22:54:39.081189", "exception": false, "start_time": "2025-07-01T22:54:38.408175", "status": "completed"}
# Import Haiku-based components
import haiku as hk
from wax.modules import EWMA as HaikuEWMA
from wax.unroll import unroll_transform_with_state

# Define Haiku-based EWMA function
@hk.transform_with_state  
def haiku_ewma_fn(x):
    return HaikuEWMA(alpha=0.1)(x)

# Initialize and apply
rng_key = jax.random.PRNGKey(42)
params, state = haiku_ewma_fn.init(rng_key, data[0])

print("Haiku - Initialized parameters:")
print(f"  logcom: {params['ewma']['logcom']}")
print("\nHaiku - Initial state keys:")
for key in state['ewma']:
    print(f"  {key}: shape {state['ewma'][key].shape}")

# %% papermill={"duration": 0.047697, "end_time": "2025-07-01T22:54:39.130414", "exception": false, "start_time": "2025-07-01T22:54:39.082717", "status": "completed"}
# Apply Haiku EWMA using unroll
haiku_unroll_fn = unroll_transform_with_state(haiku_ewma_fn)
haiku_outputs, haiku_final_state = haiku_unroll_fn.apply(
    params, state, rng_key, data
)

print(f"Haiku - Output shape: {haiku_outputs.shape}")
print(f"Haiku - Final mean value: {haiku_final_state['ewma']['mean']}")

# %% [markdown] papermill={"duration": 0.001241, "end_time": "2025-07-01T22:54:39.133072", "exception": false, "start_time": "2025-07-01T22:54:39.131831", "status": "completed"}
# ## 2. Flax Implementation
#
# New WAX-ML using Flax's explicit state management:

# %% papermill={"duration": 0.034909, "end_time": "2025-07-01T22:54:39.169164", "exception": false, "start_time": "2025-07-01T22:54:39.134255", "status": "completed"}
# Import Flax-based components
import flax.linen as nn
from wax.flax.modules import EWMA as FlaxEWMA
from wax.flax.core import flax_transform_with_state, flax_unroll_transform

# Create Flax EWMA module
flax_ewma_module = FlaxEWMA(alpha=0.1)

# Transform to init/apply functions
flax_ewma_fn = flax_transform_with_state(flax_ewma_module)

# Initialize
rng_key = jax.random.PRNGKey(42)
flax_params, flax_state = flax_ewma_fn.init(rng_key, data[0])

print("Flax - Initialized parameters:")
print(f"  logcom: {flax_params['logcom']}")
print("\nFlax - Initial state keys:")
for collection in flax_state:
    print(f"  Collection '{collection}':")
    for key in flax_state[collection]:
        print(f"    {key}: shape {flax_state[collection][key].shape}")

# %% papermill={"duration": 0.045842, "end_time": "2025-07-01T22:54:39.216490", "exception": false, "start_time": "2025-07-01T22:54:39.170648", "status": "completed"}
# Apply Flax EWMA using unroll
flax_unroll_fn = flax_unroll_transform(flax_ewma_module)
flax_outputs, flax_final_state = flax_unroll_fn.apply(
    flax_params, flax_state, rng_key, data
)

print(f"Flax - Output shape: {flax_outputs.shape}")
print(f"Flax - Final mean value: {flax_final_state['state']['mean']}")

# %% [markdown] papermill={"duration": 0.001344, "end_time": "2025-07-01T22:54:39.219392", "exception": false, "start_time": "2025-07-01T22:54:39.218048", "status": "completed"}
# ## 3. Numerical Comparison
#
# Verify that both implementations produce identical results:

# %% papermill={"duration": 0.078506, "end_time": "2025-07-01T22:54:39.299444", "exception": false, "start_time": "2025-07-01T22:54:39.220938", "status": "completed"}
# Compare outputs
output_diff = jnp.abs(haiku_outputs - flax_outputs)
max_diff = jnp.nanmax(output_diff)
mean_diff = jnp.nanmean(output_diff)

print(f"Output differences:")
print(f"  Maximum absolute difference: {max_diff}")
print(f"  Mean absolute difference: {mean_diff}")
print(f"  Outputs are numerically identical: {max_diff < 1e-10}")

# Compare final states
haiku_final_mean = haiku_final_state['ewma']['mean']
flax_final_mean = flax_final_state['state']['mean'] 
state_diff = jnp.abs(haiku_final_mean - flax_final_mean)

print(f"\nState differences:")
print(f"  Maximum state difference: {jnp.nanmax(state_diff)}")
print(f"  States are numerically identical: {jnp.nanmax(state_diff) < 1e-10}")

# %% [markdown] papermill={"duration": 0.001424, "end_time": "2025-07-01T22:54:39.302390", "exception": false, "start_time": "2025-07-01T22:54:39.300966", "status": "completed"}
# ## 4. Performance Comparison
#
# Compare compilation and execution times:

# %% papermill={"duration": 0.004665, "end_time": "2025-07-01T22:54:39.308418", "exception": false, "start_time": "2025-07-01T22:54:39.303753", "status": "completed"}
import time

def time_function(fn, *args, **kwargs):
    """Time function execution including compilation."""
    # Compile
    start = time.time()
    result = fn(*args, **kwargs)
    compile_time = time.time() - start
    
    # Execute multiple times for accurate timing
    times = []
    for _ in range(10):
        start = time.time()
        result = fn(*args, **kwargs)
        times.append(time.time() - start)
    
    return result, compile_time, np.mean(times), np.std(times)

# %% papermill={"duration": 0.445905, "end_time": "2025-07-01T22:54:39.755622", "exception": false, "start_time": "2025-07-01T22:54:39.309717", "status": "completed"}
# Time Haiku implementation
haiku_fn = lambda: haiku_unroll_fn.apply(params, state, rng_key, data)
_, haiku_compile, haiku_mean, haiku_std = time_function(haiku_fn)

print("Haiku Performance:")
print(f"  Compile time: {haiku_compile:.4f}s")
print(f"  Execution time: {haiku_mean:.4f} ± {haiku_std:.4f}s")

# %% papermill={"duration": 0.458308, "end_time": "2025-07-01T22:54:40.215648", "exception": false, "start_time": "2025-07-01T22:54:39.757340", "status": "completed"}
# Time Flax implementation  
flax_fn = lambda: flax_unroll_fn.apply(flax_params, flax_state, rng_key, data)
_, flax_compile, flax_mean, flax_std = time_function(flax_fn)

print("Flax Performance:")
print(f"  Compile time: {flax_compile:.4f}s")
print(f"  Execution time: {flax_mean:.4f} ± {flax_std:.4f}s")

print(f"\nSpeedup: {haiku_mean/flax_mean:.2f}x (Flax vs Haiku)")

# %% [markdown] papermill={"duration": 0.001444, "end_time": "2025-07-01T22:54:40.218747", "exception": false, "start_time": "2025-07-01T22:54:40.217303", "status": "completed"}
# ## 5. API Comparison
#
# Compare the developer experience and API differences:

# %% papermill={"duration": 0.00441, "end_time": "2025-07-01T22:54:40.224602", "exception": false, "start_time": "2025-07-01T22:54:40.220192", "status": "completed"}
# Haiku API Pattern
print("=== HAIKU API PATTERN ===")
print("""
# 1. Define module function
@hk.transform_with_state
def model_fn(x):
    return MyModule(params)(x)

# 2. Initialize 
params, state = model_fn.init(rng, sample_input)

# 3. Apply
output, new_state = model_fn.apply(params, state, rng, input)

# 4. State access (implicit)
# State is automatically managed by Haiku's transform system
""")

print("\n=== FLAX API PATTERN ===")
print("""
# 1. Define module class
class MyModule(nn.Module):
    def __call__(self, x):
        # Explicit state management with self.variable()
        return result

# 2. Transform for WAX-ML compatibility
model_fn = flax_transform_with_state(MyModule())

# 3. Initialize
params, state = model_fn.init(rng, sample_input)

# 4. Apply  
output, new_state = model_fn.apply(params, state, rng, input)

# 5. State access (explicit)
# State is in structured variable collections: state['collection']['var']
""")

# %% [markdown] papermill={"duration": 0.001523, "end_time": "2025-07-01T22:54:40.227506", "exception": false, "start_time": "2025-07-01T22:54:40.225983", "status": "completed"}
# ## 6. Migration Strategy
#
# For users wanting to migrate from Haiku to Flax:

# %% papermill={"duration": 0.004665, "end_time": "2025-07-01T22:54:40.233733", "exception": false, "start_time": "2025-07-01T22:54:40.229068", "status": "completed"}
print("=== MIGRATION STRATEGY ===")
print("""
1. **Parallel Adoption**:
   - Keep existing Haiku code working
   - Gradually introduce Flax modules where needed
   - Both can coexist in the same project

2. **Import Changes**:
   - `from wax.modules import EWMA` (Haiku)
   - `from wax.flax.modules import EWMA` (Flax)

3. **Transform Changes**:
   - `hk.transform_with_state(fn)` (Haiku)
   - `flax_transform_with_state(module)` (Flax)

4. **State Structure**:
   - Haiku: `state['module']['variable']`
   - Flax: `state['collection']['variable']`

5. **Benefits of Flax**:
   - More explicit state management
   - Better integration with modern JAX ecosystem  
   - Future-proofing against Haiku deprecation
   - More flexible variable collections
""")

# %% [markdown] papermill={"duration": 0.001464, "end_time": "2025-07-01T22:54:40.236743", "exception": false, "start_time": "2025-07-01T22:54:40.235279", "status": "completed"}
# ## 7. Conclusion
#
# Both implementations provide identical numerical results while offering different 
# architectural approaches:
#
# - **Haiku**: Implicit state management, familiar to existing users
# - **Flax**: Explicit state management, better ecosystem integration
#
# The parallel implementation strategy allows users to:
# 1. **Continue using existing Haiku code** without breaking changes
# 2. **Migrate gradually** to Flax for new features
# 3. **Compare approaches** side-by-side
# 4. **Future-proof** against Haiku maintenance concerns
#
# This demonstrates WAX-ML's commitment to providing both stability for existing
# users and modern alternatives for new development.

# %% papermill={"duration": 0.001521, "end_time": "2025-07-01T22:54:40.239696", "exception": false, "start_time": "2025-07-01T22:54:40.238175", "status": "completed"}
