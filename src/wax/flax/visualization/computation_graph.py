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
"""Computation graph renderer for WAX-ML streaming pipelines.

This module provides tools for visualizing the structure and dependencies
of streaming computation graphs, including:
- Pipeline topology visualization
- Module dependency analysis
- State flow tracking
- Interactive graph exploration

Key features:
- Automatic pipeline structure detection
- Multiple output formats (PNG, SVG, HTML, DOT)
- Interactive exploration with zoom/pan
- Real-time graph updates for streaming pipelines
- Integration with debugging and profiling tools
"""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp

# Optional dependencies for visualization
try:
    import graphviz

    HAS_GRAPHVIZ = True
except ImportError:
    graphviz = None
    HAS_GRAPHVIZ = False

try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import networkx as nx

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    patches = None
    nx = None
    HAS_MATPLOTLIB = False


@dataclass
class PipelineNode:
    """Represents a node in the computation pipeline."""

    id: str
    name: str
    module_type: str
    input_shapes: dict[str, tuple] = field(default_factory=dict)
    output_shapes: dict[str, tuple] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Visual attributes
    color: str = "#4CAF50"
    shape: str = "box"
    style: str = "filled"

    def __post_init__(self):
        """Assign colors based on module type."""
        color_map = {
            "EWMA": "#2196F3",
            "Buffer": "#FF9800",
            "HierarchicalStateMachine": "#9C27B0",
            "AttentionBasedStateSelector": "#E91E63",
            "CompositeStateManager": "#673AB7",
            "StreamingDebugger": "#F44336",
            "StreamingProfiler": "#795548",
            "MemoryTracker": "#607D8B",
        }

        for module_name, color in color_map.items():
            if module_name.lower() in self.module_type.lower():
                self.color = color
                break


class FlaxTransformed(Protocol):
    """Structural type of an object exposing the Flax ``init``/``apply`` pair."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def init(self, rngs: Any, *args: Any, **kwargs: Any) -> Any: ...

    def apply(self, variables: Any, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class PipelineEdge:
    """Represents an edge/connection in the computation pipeline."""

    source: str
    target: str
    label: str = ""
    data_type: str = "tensor"
    shape: tuple | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Visual attributes
    color: str = "#424242"
    style: str = "solid"
    weight: float = 1.0


class ComputationGraphRenderer:
    """Renders computation graphs for WAX-ML streaming pipelines."""

    def __init__(
        self,
        output_format: str = "png",
        include_shapes: bool = True,
        include_parameters: bool = False,
        simplify_graph: bool = True,
        max_label_length: int = 20,
    ):
        """Initialize the computation graph renderer.

        Args:
            output_format: Output format ('png', 'svg', 'pdf', 'dot', 'html')
            include_shapes: Whether to include tensor shapes in labels
            include_parameters: Whether to include module parameters
            simplify_graph: Whether to simplify complex graphs
            max_label_length: Maximum length for node labels
        """
        self.output_format = output_format
        self.include_shapes = include_shapes
        self.include_parameters = include_parameters
        self.simplify_graph = simplify_graph
        self.max_label_length = max_label_length

        self.nodes: dict[str, PipelineNode] = {}
        self.edges: list[PipelineEdge] = []
        self.graph_metadata: dict[str, Any] = {}

    def analyze_streaming_function(
        self, streaming_fn: Callable, input_example: Any, rng_key: jax.Array | None = None
    ) -> "ComputationGraphRenderer":
        """Analyze a streaming function to extract computation graph.

        Args:
            streaming_fn: The streaming function to analyze
            input_example: Example input for shape inference
            rng_key: Random key for initialization

        Returns:
            Self for method chaining
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)

        try:
            # Try to extract module structure from function
            if hasattr(streaming_fn, "init") and hasattr(streaming_fn, "apply"):
                # This is a Flax transformed function
                self._analyze_flax_function(
                    cast(FlaxTransformed, streaming_fn), input_example, rng_key
                )
            else:
                # Try to analyze as a regular function
                self._analyze_function_structure(streaming_fn, input_example)

        except Exception as e:
            warnings.warn(f"Could not fully analyze function structure: {e}", stacklevel=2)
            # Create a simple single-node graph
            self._create_simple_graph(streaming_fn, input_example)

        return self

    def _analyze_flax_function(
        self, streaming_fn: FlaxTransformed, input_example: Any, rng_key: jax.Array
    ):
        """Analyze a Flax transformed streaming function."""
        try:
            # Initialize to get parameter structure
            if isinstance(input_example, list | tuple):
                init_result = streaming_fn.init(rng_key, *input_example)
            else:
                init_result = streaming_fn.init(rng_key, input_example)

            # Handle different return formats from init
            if isinstance(init_result, tuple) and len(init_result) == 2:
                # StreamingTransform returns (params, state)
                params, state = init_result
                variables = {"params": params, "state": state}
            elif isinstance(init_result, dict):
                # Standard Flax format
                variables = init_result
            else:
                # Unknown format, treat as simple case
                variables = {"params": {}}

            # Extract module information from variables
            if "params" in variables and variables["params"]:
                self._extract_modules_from_params(variables["params"])

            # Create main function node
            main_node = PipelineNode(
                id="main",
                name=getattr(streaming_fn, "__name__", "streaming_function"),
                module_type="StreamingFunction",
                input_shapes={"input": self._get_shape(input_example)},
                parameters=self._simplify_params(variables.get("params", {})),
            )
            self.nodes["main"] = main_node

        except Exception as e:
            warnings.warn(f"Error analyzing Flax function: {e}", stacklevel=2)
            self._create_simple_graph(streaming_fn, input_example)

    def _extract_modules_from_params(self, params: dict, parent_path: str = ""):
        """Extract module structure from Flax parameters."""
        for key, value in params.items():
            node_path = f"{parent_path}.{key}" if parent_path else key

            if isinstance(value, dict):
                # This might be a nested module
                if any(
                    k in value
                    for k in ["kernel", "weight", "bias", "alpha", "scale", "logcom", "maxlen"]
                ):
                    # This looks like a leaf module with parameters
                    module_type = self._infer_module_type(key, value)

                    node = PipelineNode(
                        id=node_path,
                        name=key,
                        module_type=module_type,
                        parameters=self._simplify_params(value),
                    )
                    self.nodes[node_path] = node

                    # Add edge from parent if exists
                    if parent_path and parent_path in self.nodes:
                        edge = PipelineEdge(source=parent_path, target=node_path, label="module")
                        self.edges.append(edge)
                else:
                    # Recurse into nested structure
                    self._extract_modules_from_params(value, node_path)

    def _infer_module_type(self, name: str, params: dict) -> str:
        """Infer module type from name and parameters."""
        name_lower = name.lower()

        if "ewma" in name_lower or "alpha" in params or "logcom" in params:
            return "EWMA"
        elif "buffer" in name_lower or "maxlen" in str(params):
            return "Buffer"
        elif "attention" in name_lower or "num_heads" in str(params):
            return "AttentionBasedStateSelector"
        elif "hierarchical" in name_lower or "state_machine" in name_lower:
            return "HierarchicalStateMachine"
        elif "composite" in name_lower or "composition" in name_lower:
            return "CompositeStateManager"
        elif "kernel" in params or "weight" in params:
            return "Dense"
        else:
            return "Module"

    def _analyze_function_structure(self, fn: Callable, input_example: Any):
        """Analyze structure of a regular function."""
        # Create a single node for the function
        self._create_simple_graph(fn, input_example)

    def _create_simple_graph(self, fn: Callable, input_example: Any):
        """Create a simple single-node graph."""
        fn_name = getattr(fn, "__name__", "function")

        node = PipelineNode(
            id="main",
            name=fn_name,
            module_type="Function",
            input_shapes={"input": self._get_shape(input_example)},
        )
        self.nodes["main"] = node

    def _get_shape(self, data: Any) -> tuple:
        """Get shape of data."""
        if hasattr(data, "shape"):
            return tuple(data.shape)
        elif isinstance(data, list | tuple):
            return (len(data),)
        elif isinstance(data, dict):
            return tuple(f"{k}: {self._get_shape(v)}" for k, v in data.items())
        else:
            return ()

    def _simplify_params(self, params: dict) -> dict:
        """Simplify parameters for display."""
        if not self.include_parameters:
            return {}

        simplified: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, jnp.ndarray):
                simplified[key] = f"shape={value.shape}"
            elif isinstance(value, int | float | str | bool):
                simplified[key] = value
            elif isinstance(value, dict):
                simplified[key] = f"dict({len(value)} items)"
            else:
                simplified[key] = str(type(value).__name__)

        return simplified

    def add_node(self, node: PipelineNode) -> "ComputationGraphRenderer":
        """Add a node to the graph."""
        self.nodes[node.id] = node
        return self

    def add_edge(self, edge: PipelineEdge) -> "ComputationGraphRenderer":
        """Add an edge to the graph."""
        self.edges.append(edge)
        return self

    def render(self, output_path: str | None = None) -> str:
        """Render the computation graph.

        Args:
            output_path: Output file path (optional)

        Returns:
            Path to rendered file or HTML content
        """
        if self.output_format == "dot":
            return self._render_dot(output_path)
        elif self.output_format == "html":
            return self._render_html(output_path)
        elif HAS_GRAPHVIZ and self.output_format in ["png", "svg", "pdf"]:
            return self._render_graphviz(output_path)
        elif HAS_MATPLOTLIB:
            return self._render_matplotlib(output_path)
        else:
            return self._render_text(output_path)

    def _render_dot(self, output_path: str | None = None) -> str:
        """Render to DOT format."""
        dot_content = self._generate_dot_content()

        if output_path:
            with open(output_path, "w") as f:
                f.write(dot_content)
            return output_path
        else:
            return dot_content

    def _render_graphviz(self, output_path: str | None = None) -> str:
        """Render using Graphviz."""
        if not HAS_GRAPHVIZ:
            raise ImportError("Graphviz not available. Install with: pip install graphviz")

        dot_content = self._generate_dot_content()

        # Create graphviz object
        graph = graphviz.Source(dot_content)

        if output_path:
            # Remove extension for graphviz render
            base_path = os.path.splitext(output_path)[0]
            graph.render(base_path, format=self.output_format, cleanup=True)
            return f"{base_path}.{self.output_format}"
        else:
            # Render to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{self.output_format}", delete=False) as tmp:
                graph.render(
                    tmp.name[: -len(f".{self.output_format}")],
                    format=self.output_format,
                    cleanup=True,
                )
                return f"{tmp.name}"

    def _render_matplotlib(self, output_path: str | None = None) -> str:
        """Render using matplotlib and networkx."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib/NetworkX not available")

        # Create networkx graph
        G = nx.DiGraph()

        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(
                node_id,
                **{"label": node.name, "color": node.color, "module_type": node.module_type},
            )

        # Add edges
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, label=edge.label)

        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw nodes
        for node_id, node in self.nodes.items():
            x, y = pos[node_id]
            bbox = patches.FancyBboxPatch(
                (x - 0.1, y - 0.05),
                0.2,
                0.1,
                boxstyle="round,pad=0.01",
                facecolor=node.color,
                edgecolor="black",
                alpha=0.8,
            )
            ax.add_patch(bbox)

            # Add text
            ax.text(x, y, node.name, ha="center", va="center", fontsize=10, fontweight="bold")

        # Draw edges
        for edge in self.edges:
            if edge.source in pos and edge.target in pos:
                x1, y1 = pos[edge.source]
                x2, y2 = pos[edge.target]
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops={"arrowstyle": "->", "color": edge.color},
                )

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("WAX-ML Streaming Pipeline", fontsize=16, fontweight="bold")

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return output_path
        else:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name, dpi=300, bbox_inches="tight")
                plt.close()
                return tmp.name

    def _render_html(self, output_path: str | None = None) -> str:
        """Render to interactive HTML."""
        html_content = self._generate_html_content()

        if output_path:
            with open(output_path, "w") as f:
                f.write(html_content)
            return output_path
        else:
            return html_content

    def _render_text(self, output_path: str | None = None) -> str:
        """Render to simple text format."""
        lines = ["WAX-ML Streaming Pipeline Graph", "=" * 40, ""]

        # Nodes
        lines.append("Nodes:")
        for node_id, node in self.nodes.items():
            lines.append(f"  {node_id}: {node.name} ({node.module_type})")
            if self.include_shapes and node.input_shapes:
                lines.append(f"    Input shapes: {node.input_shapes}")
            if self.include_parameters and node.parameters:
                lines.append(f"    Parameters: {node.parameters}")

        lines.append("")

        # Edges
        lines.append("Edges:")
        for edge in self.edges:
            label = f" [{edge.label}]" if edge.label else ""
            lines.append(f"  {edge.source} -> {edge.target}{label}")

        content = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(content)
            return output_path
        else:
            return content

    def _generate_dot_content(self) -> str:
        """Generate DOT format content."""
        lines = [
            "digraph WAX_ML_Pipeline {",
            "  rankdir=LR;",
            '  node [fontname="Arial", fontsize=10];',
            '  edge [fontname="Arial", fontsize=8];',
            "",
        ]

        # Add nodes
        for node_id, node in self.nodes.items():
            label = self._create_node_label(node)
            lines.append(
                f'  "{node_id}" [label="{label}", '
                f'color="{node.color}", style="{node.style}", '
                f'shape="{node.shape}"];'
            )

        lines.append("")

        # Add edges
        for edge in self.edges:
            edge_attrs = []
            if edge.label:
                edge_attrs.append(f'label="{edge.label}"')
            if edge.color != "#424242":
                edge_attrs.append(f'color="{edge.color}"')
            if edge.style != "solid":
                edge_attrs.append(f'style="{edge.style}"')

            attrs_str = f" [{', '.join(edge_attrs)}]" if edge_attrs else ""
            lines.append(f'  "{edge.source}" -> "{edge.target}"{attrs_str};')

        lines.append("}")

        return "\n".join(lines)

    def _create_node_label(self, node: PipelineNode) -> str:
        """Create label for a node."""
        label_parts = [node.name]

        if len(node.name) > self.max_label_length:
            label_parts = [node.name[: self.max_label_length] + "..."]

        if node.module_type != node.name:
            label_parts.append(f"({node.module_type})")

        if self.include_shapes and node.input_shapes:
            shape_strs = []
            for key, shape in node.input_shapes.items():
                if isinstance(shape, tuple) and len(shape) > 0:
                    shape_strs.append(f"{key}: {shape}")
            if shape_strs:
                label_parts.append("\\n" + "\\n".join(shape_strs))

        if self.include_parameters and node.parameters:
            param_strs = []
            for key, value in list(node.parameters.items())[:3]:  # Limit to first 3
                param_strs.append(f"{key}={value}")
            if param_strs:
                label_parts.append("\\n" + "\\n".join(param_strs))

        return "\\n".join(label_parts)

    def _generate_html_content(self) -> str:
        """Generate interactive HTML content."""
        # This would create a web-based interactive visualization
        # For now, return a simple HTML representation
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>WAX-ML Pipeline Visualization</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .node {{
            border: 2px solid #333;
            margin: 10px;
            padding: 10px;
            border-radius: 5px;
            display: inline-block;
            background-color: #f0f0f0;
        }}
        .edge {{ margin: 5px 0; color: #666; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
    </style>
</head>
<body>
    <h1>WAX-ML Streaming Pipeline</h1>

    <h2>Nodes ({len(self.nodes)})</h2>
    <div>
"""

        for node_id, node in self.nodes.items():
            html += f"""
        <div class="node" style="background-color: {node.color}20;">
            <strong>{node.name}</strong> ({node.module_type})
            <br>ID: {node_id}
"""
            if self.include_shapes and node.input_shapes:
                html += f"<br>Input shapes: {node.input_shapes}"
            if self.include_parameters and node.parameters:
                html += f"<br>Parameters: {node.parameters}"
            html += "</div>"

        html += f"""
    </div>

    <h2>Edges ({len(self.edges)})</h2>
    <div>
"""

        for edge in self.edges:
            label = f" [{edge.label}]" if edge.label else ""
            html += f'<div class="edge">{edge.source} → {edge.target}{label}</div>'

        html += """
    </div>
</body>
</html>
"""
        return html


# Convenience functions


def render_pipeline_graph(
    streaming_fn: Callable,
    input_example: Any,
    output_path: str | None = None,
    format: str = "png",
    **kwargs,
) -> str:
    """Render a streaming pipeline computation graph.

    Args:
        streaming_fn: The streaming function to visualize
        input_example: Example input for shape inference
        output_path: Output file path (optional)
        format: Output format ('png', 'svg', 'pdf', 'dot', 'html')
        **kwargs: Additional arguments for ComputationGraphRenderer

    Returns:
        Path to rendered file or content
    """
    renderer = ComputationGraphRenderer(output_format=format, **kwargs)
    renderer.analyze_streaming_function(streaming_fn, input_example)
    return renderer.render(output_path)


def export_graph_to_dot(streaming_fn: Callable, input_example: Any, output_path: str) -> str:
    """Export pipeline graph to DOT format.

    Args:
        streaming_fn: The streaming function to analyze
        input_example: Example input for shape inference
        output_path: Output DOT file path

    Returns:
        Path to DOT file
    """
    return render_pipeline_graph(streaming_fn, input_example, output_path, format="dot")
