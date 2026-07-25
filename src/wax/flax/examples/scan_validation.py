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
"""Quick validation that streaming transforms work with jax.lax.scan."""

import jax
import jax.numpy as jnp

from wax.flax.core import streaming_transform_with_state
from wax.flax.modules import EWMA, Buffer


@streaming_transform_with_state
def simple_processor(x):
    """Simple streaming processor for validation."""
    buffer = Buffer(maxlen=3, fill_value=0.0)
    ema = EWMA(alpha=0.5)

    buffered = buffer(x)
    smoothed = ema(x)

    return {"input": x, "smoothed": smoothed, "mean": jnp.mean(buffered)}


def main():
    """Validate streaming transforms work with jax.lax.scan."""
    print("🔍 Streaming Transform + JAX Scan Validation")
    print("=" * 50)

    # Setup
    rng = jax.random.PRNGKey(42)
    inputs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Initialize
    params, initial_state = simple_processor.init(rng, inputs[0])

    # Method 1: For loop
    print("🔁 Method 1: For loop")
    results_loop = []
    state = initial_state
    for x in inputs:
        output, state = simple_processor.apply(params, state, None, x)
        results_loop.append(output["smoothed"])
    print(f"   Results: {[f'{x:.3f}' for x in results_loop]}")

    # Method 2: JAX scan
    print("⚡ Method 2: jax.lax.scan")

    def scan_fn(carry_state, x):
        output, new_state = simple_processor.apply(params, carry_state, None, x)
        return new_state, output["smoothed"]

    final_state, results_scan = jax.lax.scan(scan_fn, initial_state, inputs)
    print(f"   Results: {[f'{x:.3f}' for x in results_scan]}")

    # Validate
    match = jnp.allclose(jnp.array(results_loop), results_scan)
    print(f"\n✅ Results match: {match}")

    if match:
        print("🎉 VALIDATION SUCCESSFUL!")
        print("   Streaming transforms are fully compatible with jax.lax.scan")
        print("   This confirms our functional streaming architecture is JAX-native")
    else:
        print("❌ VALIDATION FAILED!")
        print("   Results differ between for-loop and scan")


if __name__ == "__main__":
    main()
