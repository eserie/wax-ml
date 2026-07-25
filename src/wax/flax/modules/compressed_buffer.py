# Copyright 2021 The WAX-ML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Memory-efficient compressed buffering for long sequences.

This module provides hierarchical buffering with multiple compression strategies
to handle arbitrarily long sequences with bounded memory usage.

The implementation is based on several key research areas and techniques:

1. **EWMA Compression**: Based on exponential smoothing literature
   - Brown, R. G. (1959). Statistical forecasting for inventory control
   - Holt, C. C. (2004). Forecasting seasonals and trends by exponentially weighted moving averages

2. **Quantile-based Compression**: Inspired by streaming quantile algorithms
   - Greenwald, M. & Khanna, S. (2001). Space-efficient online computation of quantile summaries
   - Cormode, G. & Muthukrishnan, S. (2005). An improved data stream summary: The count-min sketch

3. **Count-Min Sketch**: Probabilistic data structure for frequency estimation
   - Cormode, G. & Muthukrishnan, S. (2005). "An improved data stream summary: the count-min sketch and its applications"
   - Journal of Algorithms, 55(1), 58-75

4. **Hierarchical Memory Systems**: Inspired by computer architecture and neuroscience
   - Atkinson, R. C. & Shiffrin, R. M. (1968). Human memory: A proposed system and its control processes
   - Baddeley, A. & Hitch, G. (1974). Working memory

5. **Streaming Algorithms**: General framework for bounded-memory processing
   - Muthukrishnan, S. (2005). Data streams: Algorithms and applications
   - Babcock, B. et al. (2002). Models and issues in data stream systems

6. **Time Series Compression**: Multi-resolution representation techniques
   - Keogh, E. et al. (2001). Dimensionality reduction for fast similarity search in large time series databases
   - Yi, B. K. & Faloutsos, C. (2000). Fast time sequence indexing for arbitrary Lp norms

The hierarchical approach draws inspiration from:
- CPU cache hierarchies (L1/L2/L3 cache levels)
- Biological memory systems (sensory/short-term/long-term memory)
- Wavelets and multi-resolution analysis

Implementation optimizations for JAX/Flax:
- Pure functional design compatible with JAX transformations
- Efficient vectorized operations using jnp
- State management through Flax variable collections
- Memory usage tracking and compression ratio monitoring
"""

from collections.abc import Callable
from typing import Any, Literal

import jax.numpy as jnp
from flax import linen as nn

from ..core.streaming_transforms import streaming_transform_with_state
from .buffer import Buffer
from .ewma import EWMA

CompressionStrategy = Literal["ewma", "quantile", "downsample", "sketching", "none"]


class CompressedBuffer(nn.Module):
    """Buffer with compression for memory-efficient long sequence storage.
    
    Supports multiple compression strategies:
    - 'ewma': Exponential weighted moving average compression
    - 'quantile': Quantile-based compression maintaining key percentiles
    - 'downsample': Uniform downsampling with configurable rate
    - 'sketching': Count-Min sketch for approximate frequency tracking
    - 'none': No compression (equivalent to regular Buffer)
    
    Example:
        # High-resolution recent buffer + compressed long-term storage
        recent_buffer = Buffer(maxlen=100)  # Keep last 100 items fully
        long_term_buffer = CompressedBuffer(
            maxlen=10000, 
            compression="ewma",
            compression_params={"alpha": 0.01}
        )
    """

    maxlen: int
    compression: CompressionStrategy = "ewma"
    compression_params: dict[str, Any] | None = None
    fill_value: float = 0.0

    def setup(self):
        """Initialize compression strategy and internal storage."""
        # Default compression parameters
        default_params = {
            "ewma": {"alpha": 0.1},
            "quantile": {"percentiles": [0.1, 0.25, 0.5, 0.75, 0.9]},
            "downsample": {"factor": 2},
            "sketching": {"num_hashes": 4, "num_buckets": 1024},
            "none": {}
        }

        # Use provided params or defaults
        if self.compression not in default_params:
            raise ValueError(f"Unknown compression strategy: {self.compression}")
        params = self.compression_params or default_params[self.compression]

        # Initialize compression-specific modules
        if self.compression == "ewma":
            self._setup_ewma_compression(params)
        elif self.compression == "quantile":
            self._setup_quantile_compression(params)
        elif self.compression == "downsample":
            self._setup_downsample_compression(params)
        elif self.compression == "sketching":
            self._setup_sketching_compression(params)
        elif self.compression == "none":
            self._setup_no_compression(params)
        else:
            raise ValueError(f"Unknown compression strategy: {self.compression}")

    def _setup_ewma_compression(self, params):
        """Setup EWMA-based compression."""
        alpha = params.get("alpha", 0.1)
        self.ewma = EWMA(alpha=alpha)

        # Store compressed representation
        self.compressed_state = self.variable('state', 'compressed', lambda: 0.0)
        self.count = self.variable('state', 'count', lambda: 0)

    def _setup_quantile_compression(self, params):
        """Setup quantile-based compression."""
        percentiles = params.get("percentiles", [0.25, 0.5, 0.75])
        self.percentiles = jnp.array(percentiles)

        # Buffer for computing quantiles
        self.quantile_buffer = Buffer(maxlen=min(self.maxlen, 1000), fill_value=self.fill_value)

        # Store quantile estimates
        num_quantiles = len(percentiles)
        self.quantile_estimates = self.variable(
            'state', 'quantiles',
            lambda: jnp.full(num_quantiles, self.fill_value)
        )
        self.update_count = self.variable('state', 'update_count', lambda: 0)

    def _setup_downsample_compression(self, params):
        """Setup downsampling-based compression."""
        factor = params.get("factor", 2)
        self.downsample_factor = factor

        # Reduced-size buffer
        compressed_maxlen = max(1, self.maxlen // factor)
        self.downsampled_buffer = Buffer(maxlen=compressed_maxlen, fill_value=self.fill_value)

        # Counter for downsampling
        self.sample_counter = self.variable('state', 'sample_counter', lambda: 0)

    def _setup_sketching_compression(self, params):
        """Setup sketching-based compression."""
        num_hashes = params.get("num_hashes", 4)
        num_buckets = params.get("num_buckets", 1024)

        # Count-Min sketch structure
        self.sketch = self.variable(
            'state', 'sketch',
            lambda: jnp.zeros((num_hashes, num_buckets))
        )
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets

        # Hash parameters (simple linear congruential parameters)
        self.hash_params = self.variable(
            'state', 'hash_params',
            lambda: jnp.array([[31, 17], [37, 23], [41, 29], [43, 31]])[:num_hashes]
        )

    def _setup_no_compression(self, params):
        """Setup no compression (regular buffer)."""
        self.uncompressed_buffer = Buffer(maxlen=self.maxlen, fill_value=self.fill_value)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Add element to compressed buffer and return current state."""
        if self.compression == "ewma":
            return self._ewma_update(x)
        elif self.compression == "quantile":
            return self._quantile_update(x)
        elif self.compression == "downsample":
            return self._downsample_update(x)
        elif self.compression == "sketching":
            return self._sketching_update(x)
        elif self.compression == "none":
            return self._no_compression_update(x)

    def _ewma_update(self, x: jnp.ndarray) -> jnp.ndarray:
        """Update EWMA compressed representation."""
        # Update EWMA
        compressed_value = self.ewma(x)

        # Store in compressed state
        self.compressed_state.value = compressed_value
        self.count.value = self.count.value + 1

        return compressed_value

    def _quantile_update(self, x: jnp.ndarray) -> jnp.ndarray:
        """Update quantile-based compression."""
        # Add to quantile buffer
        buffered_values = self.quantile_buffer(x)

        # Update quantiles periodically
        update_count = self.update_count.value
        if update_count % 10 == 0:  # Update every 10 samples
            # Compute quantiles from buffer
            valid_mask = ~jnp.isnan(buffered_values)
            if jnp.sum(valid_mask) > 0:
                valid_values = buffered_values[valid_mask]
                sorted_values = jnp.sort(valid_values)
                n = len(sorted_values)

                # Compute quantile indices
                indices = self.percentiles * (n - 1)
                lower_indices = jnp.floor(indices).astype(jnp.int32)
                upper_indices = jnp.ceil(indices).astype(jnp.int32)

                # Linear interpolation for quantiles
                lower_values = sorted_values[lower_indices]
                upper_values = sorted_values[upper_indices]
                weights = indices - lower_indices

                quantiles = lower_values + weights * (upper_values - lower_values)
                self.quantile_estimates.value = quantiles

        self.update_count.value = update_count + 1

        return self.quantile_estimates.value

    def _downsample_update(self, x: jnp.ndarray) -> jnp.ndarray:
        """Update downsampled buffer."""
        counter = self.sample_counter.value

        # Only store every nth sample
        if counter % self.downsample_factor == 0:
            downsampled_data = self.downsampled_buffer(x)
        else:
            # Return current buffer state without update
            downsampled_data = self.downsampled_buffer.variables['state']['buffer']

        self.sample_counter.value = counter + 1
        return downsampled_data

    def _sketching_update(self, x: jnp.ndarray) -> jnp.ndarray:
        """Update Count-Min sketch."""
        sketch = self.sketch.value
        hash_params = self.hash_params.value

        # Hash the input value to multiple buckets
        # Simple hash function: (a * x + b) % num_buckets
        x_int = jnp.int32(x * 1000)  # Convert to integer for hashing

        # Update each hash function's bucket
        for i in range(self.num_hashes):
            a, b = hash_params[i]
            bucket = (a * x_int + b) % self.num_buckets
            sketch = sketch.at[i, bucket].add(1)

        self.sketch.value = sketch

        # Return sketch summary (min counts across hashes)
        return jnp.min(sketch, axis=0)

    def _no_compression_update(self, x: jnp.ndarray) -> jnp.ndarray:
        """Update without compression."""
        return self.uncompressed_buffer(x)

    def get_memory_usage(self) -> dict[str, int]:
        """Get estimated memory usage in bytes."""
        # Use compression params or defaults for calculations
        default_params = {
            "ewma": {"alpha": 0.1},
            "quantile": {"percentiles": [0.1, 0.25, 0.5, 0.75, 0.9]},
            "downsample": {"factor": 2},
            "sketching": {"num_hashes": 4, "num_buckets": 1024},
            "none": {}
        }
        params = self.compression_params or default_params[self.compression]

        if self.compression == "ewma":
            # Single scalar + counter
            return {"compressed_state": 8, "count": 4, "total": 12}
        elif self.compression == "quantile":
            # Quantile buffer + estimates
            percentiles = params.get("percentiles", [0.25, 0.5, 0.75])
            buffer_size = min(self.maxlen, 1000) * 8  # quantile buffer size
            estimates_size = len(percentiles) * 8
            return {
                "buffer": buffer_size,
                "estimates": estimates_size,
                "total": buffer_size + estimates_size + 4
            }
        elif self.compression == "downsample":
            # Downsampled buffer
            factor = params.get("factor", 2)
            buffer_size = (self.maxlen // factor) * 8
            return {"buffer": buffer_size, "counter": 4, "total": buffer_size + 4}
        elif self.compression == "sketching":
            # Sketch matrix + hash params
            num_hashes = params.get("num_hashes", 4)
            num_buckets = params.get("num_buckets", 1024)
            sketch_size = num_hashes * num_buckets * 8
            hash_size = num_hashes * 2 * 4
            return {"sketch": sketch_size, "hash_params": hash_size, "total": sketch_size + hash_size}
        elif self.compression == "none":
            # Full buffer
            buffer_size = self.maxlen * 8
            return {"buffer": buffer_size, "total": buffer_size}

        return {"total": 0}


class HierarchicalBuffer(nn.Module):
    """Multi-resolution hierarchical buffer for extremely long sequences.
    
    Uses multiple buffer levels with different time resolutions:
    - Recent: High-resolution, short-term (e.g., last 100 items)
    - Medium: Moderate compression, medium-term (e.g., last 1000 items compressed)
    - Long: High compression, long-term (e.g., last 10000+ items heavily compressed)
    
    Example:
        hierarchical_buffer = HierarchicalBuffer(
            recent_maxlen=100,
            medium_maxlen=1000,
            long_maxlen=10000,
            medium_compression="ewma",
            long_compression="quantile"
        )
    """

    recent_maxlen: int = 100
    medium_maxlen: int = 1000
    long_maxlen: int = 10000
    medium_compression: CompressionStrategy = "ewma"
    long_compression: CompressionStrategy = "quantile"
    fill_value: float = 0.0

    def setup(self):
        """Initialize hierarchical buffer levels."""
        # Recent buffer: no compression, high resolution
        self.recent_buffer = Buffer(maxlen=self.recent_maxlen, fill_value=self.fill_value)

        # Medium buffer: light compression
        self.medium_buffer = CompressedBuffer(
            maxlen=self.medium_maxlen,
            compression=self.medium_compression,
            compression_params={"alpha": 0.1} if self.medium_compression == "ewma" else None,
            fill_value=self.fill_value
        )

        # Long buffer: heavy compression
        self.long_buffer = CompressedBuffer(
            maxlen=self.long_maxlen,
            compression=self.long_compression,
            compression_params={"alpha": 0.01} if self.long_compression == "ewma" else None,
            fill_value=self.fill_value
        )

        # Update counters for medium/long buffer updates
        self.medium_update_counter = self.variable('state', 'medium_counter', lambda: 0)
        self.long_update_counter = self.variable('state', 'long_counter', lambda: 0)

    def __call__(self, x: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Update hierarchical buffer and return all levels."""
        # Always update recent buffer
        recent_data = self.recent_buffer(x)

        # Update medium buffer every few steps
        medium_counter = self.medium_update_counter.value
        if medium_counter % 5 == 0:  # Update every 5 steps
            medium_data = self.medium_buffer(x)
        else:
            # Return current medium buffer state
            if hasattr(self.medium_buffer, 'compressed_state'):
                medium_data = self.medium_buffer.compressed_state.value
            else:
                medium_data = jnp.array(0.0)

        # Update long buffer every many steps
        long_counter = self.long_update_counter.value
        if long_counter % 50 == 0:  # Update every 50 steps
            long_data = self.long_buffer(x)
        else:
            # Return current long buffer state
            if hasattr(self.long_buffer, 'compressed_state'):
                long_data = self.long_buffer.compressed_state.value
            else:
                long_data = jnp.array(0.0)

        # Update counters
        self.medium_update_counter.value = medium_counter + 1
        self.long_update_counter.value = long_counter + 1

        return {
            "recent": recent_data,
            "medium": medium_data,
            "long": long_data,
            "input": x
        }

    def get_total_memory_usage(self) -> dict[str, int]:
        """Get total memory usage across all levels."""
        recent_usage = self.recent_maxlen * 8  # Full precision

        # Calculate medium buffer usage based on compression strategy
        if self.medium_compression == "ewma":
            medium_usage = 12  # compressed state + count
        elif self.medium_compression == "quantile":
            medium_usage = min(self.medium_maxlen, 1000) * 8 + 5 * 8 + 4  # buffer + estimates + counter
        elif self.medium_compression == "downsample":
            medium_usage = (self.medium_maxlen // 2) * 8 + 4  # downsampled buffer + counter
        else:
            medium_usage = self.medium_maxlen * 8  # fallback

        # Calculate long buffer usage based on compression strategy
        if self.long_compression == "ewma":
            long_usage = 12  # compressed state + count
        elif self.long_compression == "quantile":
            long_usage = min(self.long_maxlen, 1000) * 8 + 5 * 8 + 4  # buffer + estimates + counter
        elif self.long_compression == "downsample":
            long_usage = (self.long_maxlen // 2) * 8 + 4  # downsampled buffer + counter
        else:
            long_usage = self.long_maxlen * 8  # fallback

        total_usage = recent_usage + medium_usage + long_usage
        uncompressed_total = (self.recent_maxlen + self.medium_maxlen + self.long_maxlen) * 8

        return {
            "recent": recent_usage,
            "medium": medium_usage,
            "long": long_usage,
            "total": total_usage,
            "compression_ratio": uncompressed_total / total_usage if total_usage > 0 else 1.0
        }


# Decorators for convenient usage

def streaming_compressed_memory(maxlen: int = 10000,
                                compression: CompressionStrategy = "ewma",
                                compression_params: dict[str, Any] | None = None):
    """Decorator for memory-efficient streaming with compression.
    
    Example:
        @streaming_compressed_memory(maxlen=100000, compression="quantile")
        def long_sequence_processor(x):
            # Process with compressed memory for very long sequences
            pass
    """
    def decorator(fn: Callable) -> Callable:
        @streaming_transform_with_state
        def wrapper(*args, **kwargs):
            compressed_buffer = CompressedBuffer(
                maxlen=maxlen,
                compression=compression,
                compression_params=compression_params
            )

            # Use first argument as input to buffer
            input_data = args[0] if args else list(kwargs.values())[0]
            buffered_data = compressed_buffer(input_data)

            # Call original function with buffered context
            return fn(buffered_data, *args, **kwargs)
        return wrapper
    return decorator


def streaming_hierarchical_memory(recent_maxlen: int = 100,
                                  medium_maxlen: int = 1000,
                                  long_maxlen: int = 10000,
                                  medium_compression: CompressionStrategy = "ewma",
                                  long_compression: CompressionStrategy = "quantile"):
    """Decorator for hierarchical memory management.
    
    Example:
        @streaming_hierarchical_memory(
            recent_maxlen=50,
            medium_maxlen=500,
            long_maxlen=5000
        )
        def multi_resolution_processor(memory_levels, x):
            recent = memory_levels["recent"]
            medium = memory_levels["medium"] 
            long_term = memory_levels["long"]
            # Process with multi-resolution memory
            pass
    """
    def decorator(fn: Callable) -> Callable:
        @streaming_transform_with_state
        def wrapper(*args, **kwargs):
            hierarchical_buffer = HierarchicalBuffer(
                recent_maxlen=recent_maxlen,
                medium_maxlen=medium_maxlen,
                long_maxlen=long_maxlen,
                medium_compression=medium_compression,
                long_compression=long_compression
            )

            # Use first argument as input to buffer
            input_data = args[0] if args else list(kwargs.values())[0]
            memory_levels = hierarchical_buffer(input_data)

            # Call original function with hierarchical memory context
            return fn(memory_levels, *args, **kwargs)
        return wrapper
    return decorator
