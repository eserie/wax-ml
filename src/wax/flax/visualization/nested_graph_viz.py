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
"""Advanced nested graph visualization for WAX-ML computation graphs.

This module provides sophisticated visualization tools for nested computational
graphs, supporting hierarchical decomposition and interactive exploration:

- Graphviz hierarchical cluster graphs with subgraphs
- Cytoscape.js interactive nested graph visualization
- D3.js force-directed hierarchical layouts
- Interactive graph exploration with zoom/collapse
- Modern graph layout algorithms for complex structures
- Real-time graph updates for streaming computations

Key features:
- Hierarchical module decomposition with automatic nesting
- Interactive exploration with expand/collapse functionality
- High-performance rendering for large computation graphs
- Export capabilities for various formats (SVG, PNG, HTML, JSON)
- Integration with existing WAX-ML streaming architecture
- Responsive design optimized for modern browsers
"""

from __future__ import annotations

import json
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# Graphviz imports with fallback
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    graphviz = None
    HAS_GRAPHVIZ = False

# NetworkX for graph analysis
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False

# Plotly for interactive graphs
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None
    make_subplots = None
    HAS_PLOTLY = False

# IPython for display
try:
    import IPython.display as ipython_display
    HAS_IPYTHON = True
except ImportError:
    ipython_display = None
    HAS_IPYTHON = False

from .computation_graph import ComputationGraphRenderer, PipelineEdge, PipelineNode


@dataclass
class GraphHierarchy:
    """Represents hierarchical structure of computation graph."""

    node_id: str
    name: str
    node_type: str
    level: int
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def add_child(self, child_id: str) -> None:
        """Add a child node to this hierarchy node."""
        if child_id not in self.children:
            self.children.append(child_id)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0


@dataclass
class NestedGraphConfig:
    """Configuration for nested graph visualization."""

    # Graphviz configuration
    graphviz_engine: str = "dot"  # dot, neato, fdp, sfdp, circo, twopi
    graphviz_format: str = "svg"  # svg, png, pdf, dot
    graphviz_dpi: int = 300

    # Hierarchical layout
    max_nesting_levels: int = 5
    cluster_min_nodes: int = 2
    auto_cluster: bool = True

    # Visual styling
    node_colors: dict[str, str] = field(default_factory=lambda: {
        "module": "#4CAF50",
        "function": "#2196F3",
        "buffer": "#FF9800",
        "ewma": "#9C27B0",
        "state": "#607D8B",
        "input": "#FFC107",
        "output": "#F44336"
    })

    cluster_colors: list[str] = field(default_factory=lambda: [
        "#E3F2FD", "#F3E5F5", "#E8F5E8", "#FFF8E1", "#FCE4EC", "#F1F8E9"
    ])

    # Interactive features
    enable_zoom: bool = True
    enable_collapse: bool = True
    enable_tooltips: bool = True

    # Export options
    include_metadata: bool = True
    compress_output: bool = False


class HierarchicalGraphAnalyzer:
    """Analyzes computation graphs to extract hierarchical structure."""

    def __init__(self, config: NestedGraphConfig = None):
        self.config = config or NestedGraphConfig()
        self.hierarchy: dict[str, GraphHierarchy] = {}
        self.levels: dict[int, list[str]] = defaultdict(list)

    def analyze_streaming_function(self, streaming_fn: Any, input_example: Any) -> dict[str, GraphHierarchy]:
        """Analyze streaming function to extract hierarchical structure.

        Args:
            streaming_fn: The streaming function to analyze
            input_example: Example input for analysis

        Returns:
            Dictionary mapping node IDs to hierarchy information
        """
        # Use existing computation graph renderer
        renderer = ComputationGraphRenderer()
        renderer.analyze_streaming_function(streaming_fn, input_example)

        # Clear previous analysis
        self.hierarchy.clear()
        self.levels.clear()

        # Extract hierarchical structure from nodes
        self._extract_hierarchy_from_nodes(renderer.nodes, renderer.edges)

        # Auto-cluster related nodes if enabled
        if self.config.auto_cluster:
            self._auto_cluster_nodes()

        # Assign levels based on hierarchy
        self._assign_hierarchy_levels()

        return self.hierarchy

    def _extract_hierarchy_from_nodes(self, nodes: dict[str, PipelineNode], edges: list[PipelineEdge]) -> None:
        """Extract hierarchy from computation graph nodes."""
        # Create hierarchy nodes from pipeline nodes
        for node_id, node in nodes.items():
            # Determine node type and potential parent
            node_type = self._classify_node_type(node)
            parent_id = self._infer_parent_relationship(node_id, node, nodes)

            hierarchy_node = GraphHierarchy(
                node_id=node_id,
                name=node.name,
                node_type=node_type,
                level=0,  # Will be assigned later
                parent=parent_id,
                properties={
                    "module_type": node.module_type,
                    "parameters": node.parameters,
                    "input_shapes": node.input_shapes,
                    "output_shapes": node.output_shapes
                }
            )

            self.hierarchy[node_id] = hierarchy_node

        # Build parent-child relationships
        for node_id, hierarchy_node in self.hierarchy.items():
            if hierarchy_node.parent and hierarchy_node.parent in self.hierarchy:
                self.hierarchy[hierarchy_node.parent].add_child(node_id)

    def _classify_node_type(self, node: PipelineNode) -> str:
        """Classify node type based on module information."""
        module_type = node.module_type.lower()

        if "ewma" in module_type:
            return "ewma"
        elif "buffer" in module_type:
            return "buffer"
        elif "state" in module_type:
            return "state"
        elif "function" in module_type:
            return "function"
        elif "module" in module_type:
            return "module"
        else:
            return "unknown"

    def _infer_parent_relationship(
        self, node_id: str, node: PipelineNode, all_nodes: dict[str, PipelineNode]
    ) -> str | None:
        """Infer parent relationship based on naming patterns and structure."""
        # Look for hierarchical naming patterns (e.g., "module.submodule")
        if "." in node_id:
            parent_parts = node_id.split(".")[:-1]
            potential_parent = ".".join(parent_parts)
            if potential_parent in all_nodes:
                return potential_parent

        # Look for module containers
        for other_id, other_node in all_nodes.items():
            if (
                other_id != node_id
                and other_node.module_type == "Module"
                and node_id.startswith(other_id)
            ):
                return other_id

        return None

    def _auto_cluster_nodes(self) -> None:
        """Automatically cluster related nodes into hierarchical groups."""
        # Group nodes by type and connectivity
        type_groups = defaultdict(list)
        for node_id, hierarchy in self.hierarchy.items():
            if hierarchy.parent is None:  # Only cluster root nodes
                type_groups[hierarchy.node_type].append(node_id)

        # Create cluster nodes for groups with sufficient size
        cluster_id = 0
        for node_type, node_ids in type_groups.items():
            if len(node_ids) >= self.config.cluster_min_nodes:
                cluster_name = f"cluster_{node_type}_{cluster_id}"

                # Create cluster node
                cluster_node = GraphHierarchy(
                    node_id=cluster_name,
                    name=f"{node_type.title()} Components",
                    node_type="cluster",
                    level=0,
                    properties={"cluster_type": node_type, "member_count": len(node_ids)}
                )

                self.hierarchy[cluster_name] = cluster_node

                # Assign nodes to cluster
                for node_id in node_ids:
                    self.hierarchy[node_id].parent = cluster_name
                    cluster_node.add_child(node_id)

                cluster_id += 1

    def _assign_hierarchy_levels(self) -> None:
        """Assign hierarchy levels based on parent-child relationships."""
        # Find root nodes (no parents)
        roots = [node_id for node_id, h in self.hierarchy.items() if h.parent is None]

        # BFS traversal to assign levels
        queue = [(node_id, 0) for node_id in roots]

        while queue:
            node_id, level = queue.pop(0)

            if node_id in self.hierarchy:
                self.hierarchy[node_id].level = level
                self.levels[level].append(node_id)

                # Add children to queue
                for child_id in self.hierarchy[node_id].children:
                    queue.append((child_id, level + 1))

    def get_max_level(self) -> int:
        """Get the maximum hierarchy level."""
        return max(self.levels.keys()) if self.levels else 0


class GraphvizNestedRenderer:
    """Renders nested graphs using Graphviz with hierarchical clusters."""

    def __init__(self, config: NestedGraphConfig = None):
        self.config = config or NestedGraphConfig()

        if not HAS_GRAPHVIZ:
            warnings.warn(
                "Graphviz not available. Install with: pip install graphviz", stacklevel=2
            )

    def render_hierarchical_graph(
        self, hierarchy: dict[str, GraphHierarchy], output_path: str | None = None
    ) -> str:
        """Render hierarchical graph using Graphviz clusters.

        Args:
            hierarchy: Hierarchical graph structure
            output_path: Optional output file path

        Returns:
            Path to rendered file or dot source
        """
        if not HAS_GRAPHVIZ:
            raise ImportError("Graphviz required for hierarchical rendering")

        # Create main graph
        dot = graphviz.Digraph(
            engine=self.config.graphviz_engine,
            format=self.config.graphviz_format
        )

        # Set graph attributes
        dot.attr(
            rankdir="TB",
            dpi=str(self.config.graphviz_dpi),
            bgcolor="white",
            fontname="Arial",
            fontsize="12"
        )

        # Render hierarchical structure
        self._render_hierarchy_recursive(dot, hierarchy, None, 0)

        # Add edges between nodes
        self._add_hierarchy_edges(dot, hierarchy)

        # Render to file or return source
        if output_path:
            dot.render(output_path, cleanup=True)
            return f"{output_path}.{self.config.graphviz_format}"
        else:
            return str(dot.source)

    def _render_hierarchy_recursive(
        self,
        dot: graphviz.Digraph,
        hierarchy: dict[str, GraphHierarchy],
        parent_id: str | None,
        level: int,
    ) -> None:
        """Recursively render hierarchy with clusters."""
        # Get nodes at current level with specified parent
        current_nodes = [
            h for h in hierarchy.values()
            if h.parent == parent_id and h.level == level
        ]

        for node in current_nodes:
            if node.children:
                # Create cluster for parent node
                cluster_name = f"cluster_{node.node_id}"
                cluster_color = self.config.cluster_colors[level % len(self.config.cluster_colors)]

                with dot.subgraph(name=cluster_name) as cluster:
                    cluster.attr(
                        style="filled",
                        fillcolor=cluster_color,
                        label=node.name,
                        fontsize="14",
                        fontweight="bold"
                    )

                    # Add the parent node itself
                    self._add_node_to_graph(cluster, node)

                    # Recursively add children
                    self._render_hierarchy_recursive(cluster, hierarchy, node.node_id, level + 1)
            else:
                # Leaf node - add directly
                self._add_node_to_graph(dot, node)

    def _add_node_to_graph(self, graph: graphviz.Digraph, node: GraphHierarchy) -> None:
        """Add a single node to the graph."""
        node_color = self.config.node_colors.get(node.node_type, "#CCCCCC")

        # Create node label with details
        label_parts = [node.name]
        if self.config.include_metadata and node.properties:
            if "parameters" in node.properties and node.properties["parameters"]:
                params = node.properties["parameters"]
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                label_parts.append(f"({param_str})")

        label = "\\n".join(label_parts)

        graph.node(
            node.node_id,
            label=label,
            style="filled",
            fillcolor=node_color,
            shape="box" if node.node_type == "module" else "ellipse",
            fontname="Arial",
            fontsize="10"
        )

    def _add_hierarchy_edges(self, dot: graphviz.Digraph, hierarchy: dict[str, GraphHierarchy]) -> None:
        """Add edges between hierarchy nodes."""
        # For now, add parent-child edges
        for node in hierarchy.values():
            for child_id in node.children:
                if child_id in hierarchy:
                    dot.edge(
                        node.node_id,
                        child_id,
                        style="dashed",
                        color="gray",
                        arrowhead="open"
                    )


class CytoscapeNestedRenderer:
    """Renders interactive nested graphs using Cytoscape.js."""

    def __init__(self, config: NestedGraphConfig = None):
        self.config = config or NestedGraphConfig()

    def render_interactive_graph(self, hierarchy: dict[str, GraphHierarchy]) -> str:
        """Render interactive nested graph using Cytoscape.js.

        Args:
            hierarchy: Hierarchical graph structure

        Returns:
            HTML content with embedded Cytoscape.js visualization
        """
        # Convert hierarchy to Cytoscape format
        cytoscape_data = self._convert_to_cytoscape_format(hierarchy)

        # Generate unique container ID
        container_id = f"cytoscape-{uuid.uuid4().hex[:8]}"

        # Create HTML with Cytoscape.js
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>WAX-ML Nested Graph Visualization</title>
    <script src="https://unpkg.com/cytoscape@3.21.0/dist/cytoscape.min.js"></script>
    <script src="https://unpkg.com/cytoscape-compound-drag-and-drop@1.0.0/cytoscape-compound-drag-and-drop.js"></script>
    <script src="https://unpkg.com/cytoscape-cola@2.5.1/cytoscape-cola.js"></script>
    <style>
        #{container_id} {{
            width: 100%;
            height: 600px;
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .graph-controls {{
            margin: 10px 0;
            text-align: center;
        }}
        .control-button {{
            margin: 0 5px;
            padding: 8px 16px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .control-button:hover {{
            background: #1976D2;
        }}
    </style>
</head>
<body>
    <div class="graph-controls">
        <button class="control-button" onclick="expandAll()">Expand All</button>
        <button class="control-button" onclick="collapseAll()">Collapse All</button>
        <button class="control-button" onclick="resetLayout()">Reset Layout</button>
        <button class="control-button" onclick="fitToScreen()">Fit to Screen</button>
    </div>

    <div id="{container_id}"></div>

    <script>
        const cy = cytoscape({{
            container: document.getElementById('{container_id}'),

            elements: {json.dumps(cytoscape_data, indent=2)},

            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'background-color': 'data(color)',
                        'border-width': 2,
                        'border-color': '#333',
                        'font-size': '12px',
                        'font-family': 'Arial, sans-serif',
                        'width': 'label',
                        'height': 'label',
                        'padding': '10px'
                    }}
                }},
                {{
                    selector: 'node[type="cluster"]',
                    style: {{
                        'background-color': 'data(color)',
                        'background-opacity': 0.3,
                        'border-width': 3,
                        'border-color': 'data(color)',
                        'border-opacity': 0.8,
                        'font-weight': 'bold',
                        'font-size': '14px'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#666',
                        'target-arrow-color': '#666',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }},
                {{
                    selector: 'edge[type="hierarchy"]',
                    style: {{
                        'line-style': 'dashed',
                        'line-color': '#999',
                        'target-arrow-color': '#999'
                    }}
                }},
                {{
                    selector: ':selected',
                    style: {{
                        'border-width': 4,
                        'border-color': '#FF5722'
                    }}
                }}
            ],

            layout: {{
                name: 'cola',
                animate: true,
                refresh: 1,
                maxSimulationTime: 2000,
                ungrabifyWhileSimulating: false,
                fit: true,
                padding: 30,
                nodeDimensionsIncludeLabels: true,
                randomize: false,
                avoidOverlap: true,
                handleDisconnected: true,
                convergenceThreshold: 0.01,
                nodeSpacing: function(node) {{ return 10; }},
                flow: {{ axis: 'y', minSeparation: 30 }}
            }},

            wheelSensitivity: 0.1,
            minZoom: 0.1,
            maxZoom: 3
        }});

        // Event handlers
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            if (node.data('type') === 'cluster') {{
                // Toggle collapse/expand
                const children = node.children();
                if (children.length > 0) {{
                    if (children.hidden()) {{
                        children.show();
                        node.data('expanded', true);
                    }} else {{
                        children.hide();
                        node.data('expanded', false);
                    }}
                    cy.layout({{name: 'cola', animate: true}}).run();
                }}
            }}
        }});

        // Add tooltips
        cy.on('mouseover', 'node', function(evt) {{
            const node = evt.target;
            const properties = node.data('properties');
            if (properties) {{
                node.popover({{
                    content: JSON.stringify(properties, null, 2),
                    position: 'top'
                }});
            }}
        }});

        // Control functions
        function expandAll() {{
            cy.nodes('[type="cluster"]').forEach(function(node) {{
                node.children().show();
                node.data('expanded', true);
            }});
            cy.layout({{name: 'cola', animate: true}}).run();
        }}

        function collapseAll() {{
            cy.nodes('[type="cluster"]').forEach(function(node) {{
                node.children().hide();
                node.data('expanded', false);
            }});
            cy.layout({{name: 'cola', animate: true}}).run();
        }}

        function resetLayout() {{
            cy.layout({{name: 'cola', animate: true, randomize: true}}).run();
        }}

        function fitToScreen() {{
            cy.fit();
        }}
    </script>
</body>
</html>
        """

        return html_content

    def _convert_to_cytoscape_format(self, hierarchy: dict[str, GraphHierarchy]) -> list[dict[str, Any]]:
        """Convert hierarchy to Cytoscape.js format."""
        elements: list[dict[str, Any]] = []

        # Add nodes
        for node in hierarchy.values():
            node_color = self.config.node_colors.get(node.node_type, "#CCCCCC")

            element = {
                "data": {
                    "id": node.node_id,
                    "label": node.name,
                    "type": node.node_type,
                    "color": node_color,
                    "properties": node.properties,
                    "level": node.level
                }
            }

            # Set parent for nested structure
            if node.parent:
                element["data"]["parent"] = node.parent

            elements.append(element)

        # Add edges
        for node in hierarchy.values():
            for child_id in node.children:
                edge = {
                    "data": {
                        "id": f"{node.node_id}_{child_id}",
                        "source": node.node_id,
                        "target": child_id,
                        "type": "hierarchy"
                    }
                }
                elements.append(edge)

        return elements


class D3NestedRenderer:
    """Renders nested graphs using D3.js force-directed layout."""

    def __init__(self, config: NestedGraphConfig = None):
        self.config = config or NestedGraphConfig()

    def render_d3_graph(self, hierarchy: dict[str, GraphHierarchy]) -> str:
        """Render nested graph using D3.js force-directed layout.

        Args:
            hierarchy: Hierarchical graph structure

        Returns:
            HTML content with embedded D3.js visualization
        """
        # Convert hierarchy to D3 format
        d3_data = self._convert_to_d3_format(hierarchy)

        # Generate unique container ID
        container_id = f"d3-graph-{uuid.uuid4().hex[:8]}"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>WAX-ML D3 Nested Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        #{container_id} {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: #fafafa;
        }}
        .node {{
            cursor: pointer;
            stroke: #333;
            stroke-width: 2;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 2;
        }}
        .cluster {{
            fill: none;
            stroke: #000;
            stroke-dasharray: 5,5;
            stroke-width: 2;
        }}
        .tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            font-size: 12px;
            max-width: 200px;
        }}
    </style>
</head>
<body>
    <div id="{container_id}"></div>

    <script>
        const data = {json.dumps(d3_data, indent=2)};

        const width = 800;
        const height = 600;

        const svg = d3.select('#{container_id}')
            .append('svg')
            .attr('width', width)
            .attr('height', height);

        // Create tooltip
        const tooltip = d3.select('body')
            .append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);

        // Create force simulation
        const simulation = d3.forceSimulation(data.nodes)
            .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));

        // Add links
        const link = svg.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(data.links)
            .enter().append('line')
            .attr('class', 'link');

        // Add nodes
        const node = svg.append('g')
            .attr('class', 'nodes')
            .selectAll('circle')
            .data(data.nodes)
            .enter().append('circle')
            .attr('class', 'node')
            .attr('r', d => d.type === 'cluster' ? 20 : 15)
            .attr('fill', d => d.color)
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        // Add labels
        const label = svg.append('g')
            .attr('class', 'labels')
            .selectAll('text')
            .data(data.nodes)
            .enter().append('text')
            .text(d => d.name)
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em')
            .style('font-size', '10px')
            .style('font-family', 'Arial, sans-serif');

        // Add hover effects
        node.on('mouseover', function(event, d) {{
            tooltip.transition()
                .duration(200)
                .style('opacity', .9);
            tooltip.html(
                `<strong>${{d.name}}</strong><br/>
                Type: ${{d.type}}<br/>
                Level: ${{d.level}}<br/>
                Properties: ${{JSON.stringify(d.properties, null, 2)}}`
            )
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        }})
        .on('mouseout', function(d) {{
            tooltip.transition()
                .duration(500)
                .style('opacity', 0);
        }});

        // Update positions on simulation tick
        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        }});

        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
        """

        return html_content

    def _convert_to_d3_format(self, hierarchy: dict[str, GraphHierarchy]) -> dict[str, Any]:
        """Convert hierarchy to D3.js format."""
        nodes = []
        links = []

        # Add nodes
        for node in hierarchy.values():
            node_color = self.config.node_colors.get(node.node_type, "#CCCCCC")

            d3_node = {
                "id": node.node_id,
                "name": node.name,
                "type": node.node_type,
                "level": node.level,
                "color": node_color,
                "properties": node.properties
            }
            nodes.append(d3_node)

        # Add links
        for node in hierarchy.values():
            for child_id in node.children:
                link = {
                    "source": node.node_id,
                    "target": child_id,
                    "type": "hierarchy"
                }
                links.append(link)

        return {"nodes": nodes, "links": links}


class NestedGraphVisualizer:
    """Main class for nested graph visualization with multiple backends."""

    def __init__(self, config: NestedGraphConfig = None):
        self.config = config or NestedGraphConfig()
        self.analyzer = HierarchicalGraphAnalyzer(self.config)
        self.graphviz_renderer = GraphvizNestedRenderer(self.config)
        self.cytoscape_renderer = CytoscapeNestedRenderer(self.config)
        self.d3_renderer = D3NestedRenderer(self.config)

    def visualize_computation_graph(
        self,
        streaming_fn: Any,
        input_example: Any,
        backend: str = "graphviz",
        output_path: str | None = None,
    ) -> str:
        """Visualize computation graph with nested structure.

        Args:
            streaming_fn: Streaming function to visualize
            input_example: Example input for analysis
            backend: Visualization backend ('graphviz', 'cytoscape', 'd3')
            output_path: Optional output file path

        Returns:
            Path to output file or HTML content
        """
        # Analyze hierarchical structure
        hierarchy = self.analyzer.analyze_streaming_function(streaming_fn, input_example)

        if backend == "graphviz":
            return self.graphviz_renderer.render_hierarchical_graph(hierarchy, output_path)
        elif backend == "cytoscape":
            html_content = self.cytoscape_renderer.render_interactive_graph(hierarchy)
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(html_content)
                return output_path
            return html_content
        elif backend == "d3":
            html_content = self.d3_renderer.render_d3_graph(hierarchy)
            if output_path:
                with open(output_path, 'w') as f:
                    f.write(html_content)
                return output_path
            return html_content
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def display_in_jupyter(
        self, streaming_fn: Any, input_example: Any, backend: str = "cytoscape"
    ) -> None:
        """Display nested graph in Jupyter notebook.

        Args:
            streaming_fn: Streaming function to visualize
            input_example: Example input for analysis
            backend: Visualization backend ('graphviz', 'cytoscape', 'd3')
        """
        if not HAS_IPYTHON:
            print("IPython not available for display")
            return

        result = self.visualize_computation_graph(streaming_fn, input_example, backend)

        if backend == "graphviz" and result.endswith('.svg'):
            # Display SVG directly
            with open(result) as f:
                svg_content = f.read()
            ipython_display.display(ipython_display.SVG(svg_content))
        else:
            # Display HTML content
            ipython_display.display(ipython_display.HTML(result))

    def get_hierarchy_summary(self, streaming_fn: Any, input_example: Any) -> dict[str, Any]:
        """Get summary of hierarchical structure.

        Args:
            streaming_fn: Streaming function to analyze
            input_example: Example input for analysis

        Returns:
            Summary of hierarchy including statistics
        """
        hierarchy = self.analyzer.analyze_streaming_function(streaming_fn, input_example)

        # Calculate statistics
        total_nodes = len(hierarchy)
        max_level = self.analyzer.get_max_level()
        node_types: defaultdict[str, int] = defaultdict(int)
        nodes_per_level: defaultdict[int, int] = defaultdict(int)

        for node in hierarchy.values():
            node_types[node.node_type] += 1
            nodes_per_level[node.level] += 1

        return {
            "total_nodes": total_nodes,
            "max_hierarchy_level": max_level,
            "node_types": dict(node_types),
            "nodes_per_level": dict(nodes_per_level),
            "hierarchy": hierarchy
        }


# Convenience functions
def visualize_nested_graph(
    streaming_fn: Any,
    input_example: Any,
    backend: str = "cytoscape",
    output_path: str | None = None,
    config: NestedGraphConfig = None,
) -> str:
    """Convenience function for nested graph visualization.

    Args:
        streaming_fn: Streaming function to visualize
        input_example: Example input for analysis
        backend: Visualization backend ('graphviz', 'cytoscape', 'd3')
        output_path: Optional output file path
        config: Optional configuration

    Returns:
        Path to output file or HTML content
    """
    visualizer = NestedGraphVisualizer(config)
    return visualizer.visualize_computation_graph(streaming_fn, input_example, backend, output_path)


def display_nested_graph_jupyter(
    streaming_fn: Any,
    input_example: Any,
    backend: str = "cytoscape",
    config: NestedGraphConfig = None,
) -> None:
    """Display nested graph in Jupyter notebook.

    Args:
        streaming_fn: Streaming function to visualize
        input_example: Example input for analysis
        backend: Visualization backend ('graphviz', 'cytoscape', 'd3')
        config: Optional configuration
    """
    visualizer = NestedGraphVisualizer(config)
    visualizer.display_in_jupyter(streaming_fn, input_example, backend)
