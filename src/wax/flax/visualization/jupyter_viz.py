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
"""Modern interactive visualization tools for Jupyter notebooks.

This module provides state-of-the-art visualization capabilities specifically
designed for Jupyter notebook environments, including:
- Interactive Plotly charts with real-time updates
- Bokeh streaming visualizations
- ipywidgets interactive controls
- Animated pipeline flow diagrams
- 3D data visualization
- Interactive parameter tuning interfaces

Key features:
- Native Jupyter integration with rich display capabilities
- Real-time streaming data visualization with smooth animations
- Interactive parameter controls with immediate visual feedback
- Modern web-based visualization backends (Plotly, Bokeh)
- Export capabilities for presentations and reports
- Mobile-responsive design for modern workflows
"""

from __future__ import annotations

import threading
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Core visualization dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.graph_objs import FigureWidget
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    pyo = None
    FigureWidget = None
    HAS_PLOTLY = False

try:
    import bokeh.palettes as bp
    from bokeh.io import push_notebook
    from bokeh.layouts import column, row
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure, output_notebook, show

    HAS_BOKEH = True
except ImportError:
    figure = None
    show = None
    output_notebook = None
    ColumnDataSource = None
    HoverTool = None
    column = None
    row = None
    push_notebook = None
    bp = None
    HAS_BOKEH = False

try:
    import ipywidgets as widgets
    from IPython.display import display

    HAS_WIDGETS = True
except ImportError:
    widgets = None
    display = None
    HAS_WIDGETS = False

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False

from .computation_graph import ComputationGraphRenderer
from .data_flow_visualizer import DataFlowTracker


@dataclass
class JupyterVizConfig:
    """Configuration for Jupyter visualization components."""

    # Plotly configuration
    plotly_theme: str = "plotly_white"
    plotly_height: int = 500
    plotly_width: int = 800
    show_toolbar: bool = True

    # Bokeh configuration
    bokeh_width: int = 800
    bokeh_height: int = 400
    bokeh_tools: str = "pan,wheel_zoom,box_zoom,reset,save"

    # Animation configuration
    animation_interval_ms: int = 100
    max_history_points: int = 1000

    # Color schemes
    color_palette: list[str] = field(
        default_factory=lambda: [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )

    # Interactive controls
    enable_widgets: bool = True
    widget_layout_width: str = "400px"


class InteractivePipelineGraph:
    """Interactive pipeline graph visualization using Plotly and NetworkX."""

    def __init__(self, config: JupyterVizConfig = None):
        self.config = config or JupyterVizConfig()

        if not HAS_PLOTLY:
            warnings.warn("Plotly not available. Install with: pip install plotly", stacklevel=2)
        if not HAS_NETWORKX:
            warnings.warn(
                "NetworkX not available. Install with: pip install networkx", stacklevel=2
            )

    def create_pipeline_graph(
        self, streaming_fn: Any, input_example: Any, layout: str = "spring"
    ) -> FigureWidget:
        """Create an interactive pipeline graph visualization.

        Args:
            streaming_fn: The streaming function to visualize
            input_example: Example input for analysis
            layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')

        Returns:
            Interactive Plotly FigureWidget
        """
        if not HAS_PLOTLY or not HAS_NETWORKX:
            raise ImportError("Plotly and NetworkX required for interactive graphs")

        # Analyze the pipeline structure
        renderer = ComputationGraphRenderer()
        renderer.analyze_streaming_function(streaming_fn, input_example)

        # Create NetworkX graph
        G = nx.DiGraph()

        # Add nodes
        for node_id, node in renderer.nodes.items():
            G.add_node(node_id, name=node.name, type=node.module_type, params=str(node.parameters))

        # Add edges
        for edge in renderer.edges:
            G.add_edge(edge.source, edge.target, label=edge.label)

        # Generate layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Extract positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]

        # Create edge traces
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge[0]} → {edge[1]}")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, line={"width": 2, "color": "#888"}, hoverinfo="none", mode="lines"
        )

        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=[G.nodes[node]["name"] for node in G.nodes()],
            textposition="middle center",
            hovertext=[
                f"<b>{G.nodes[node]['name']}</b><br>"
                f"Type: {G.nodes[node]['type']}<br>"
                f"Parameters: {G.nodes[node]['params']}"
                for node in G.nodes()
            ],
            marker={
                "showscale": True,
                "colorscale": "Viridis",
                "reversescale": True,
                "color": [],
                "size": 30,
                "colorbar": {"thickness": 15, "len": 0.5, "x": 1.02, "title": "Node Depth"},
                "line": {"width": 2},
            },
        )

        # Color nodes by depth
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))

        node_trace.marker.color = node_adjacencies

        # Create figure
        fig = go.FigureWidget(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Interactive Pipeline Graph",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin={"b": 20, "l": 5, "r": 5, "t": 40},
                annotations=[
                    {
                        "text": "Hover over nodes for details. Drag to pan, scroll to zoom.",
                        "showarrow": False,
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.005,
                        "y": -0.002,
                        "xanchor": "left",
                        "yanchor": "bottom",
                        "font": {"color": "#888", "size": 12},
                    }
                ],
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                plot_bgcolor="white",
                height=self.config.plotly_height,
                width=self.config.plotly_width,
            ),
        )

        return fig


class StreamingDataVisualizer:
    """Real-time streaming data visualization with Plotly."""

    def __init__(self, config: JupyterVizConfig = None):
        self.config = config or JupyterVizConfig()
        self.figures: dict[str, FigureWidget] = {}
        self.data_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_history_points)
        )
        self.time_buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_history_points)
        )
        self.is_streaming = False
        self._update_thread = None

        if not HAS_PLOTLY:
            warnings.warn("Plotly not available. Install with: pip install plotly", stacklevel=2)

    def create_streaming_plot(
        self, plot_type: str = "line", title: str = "Streaming Data", y_label: str = "Value"
    ) -> FigureWidget:
        """Create a streaming plot that updates in real-time.

        Args:
            plot_type: Type of plot ('line', 'scatter', 'bar')
            title: Plot title
            y_label: Y-axis label

        Returns:
            Interactive Plotly FigureWidget
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly required for streaming plots")

        fig = go.FigureWidget()

        # Configure layout for streaming
        fig.update_layout(
            title=title,
            xaxis_title="Time Step",
            yaxis_title=y_label,
            height=self.config.plotly_height,
            width=self.config.plotly_width,
            template=self.config.plotly_theme,
            showlegend=True,
            hovermode="x unified",
        )

        # Store figure
        fig_id = f"stream_{len(self.figures)}"
        self.figures[fig_id] = fig

        return fig

    def add_stream(self, fig: FigureWidget, stream_name: str, color: str = None) -> None:
        """Add a data stream to the plot.

        Args:
            fig: The figure to add stream to
            stream_name: Name of the data stream
            color: Color for the stream line
        """
        if color is None:
            color_idx = len(fig.data) % len(self.config.color_palette)
            color = self.config.color_palette[color_idx]

        fig.add_scatter(
            x=[],
            y=[],
            mode="lines+markers",
            name=stream_name,
            line={"color": color, "width": 2},
            marker={"size": 4},
        )

    def update_stream(
        self, fig: FigureWidget, stream_name: str, value: float, timestamp: float = None
    ) -> None:
        """Update a data stream with new value.

        Args:
            fig: The figure containing the stream
            stream_name: Name of the stream to update
            value: New data value
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = time.time()

        # Find the trace for this stream
        trace_idx = None
        for i, trace in enumerate(fig.data):
            if trace.name == stream_name:
                trace_idx = i
                break

        if trace_idx is None:
            warnings.warn(f"Stream '{stream_name}' not found in figure", stacklevel=2)
            return

        # Update buffers
        buffer_key = f"{id(fig)}_{stream_name}"
        self.data_buffers[buffer_key].append(value)
        self.time_buffers[buffer_key].append(timestamp)

        # Update trace
        with fig.batch_update():
            fig.data[trace_idx].x = list(self.time_buffers[buffer_key])
            fig.data[trace_idx].y = list(self.data_buffers[buffer_key])


class InteractiveParameterControls:
    """Interactive parameter controls using ipywidgets."""

    def __init__(self, config: JupyterVizConfig = None):
        self.config = config or JupyterVizConfig()
        self.controls: dict[str, Any] = {}
        self.callbacks: list[Callable[[dict[str, Any]], None]] = []

        if not HAS_WIDGETS:
            warnings.warn(
                "ipywidgets not available. Install with: pip install ipywidgets", stacklevel=2
            )

    def create_parameter_panel(self, parameters: dict[str, dict[str, Any]]) -> widgets.VBox:
        """Create an interactive parameter control panel.

        Args:
            parameters: Dict mapping parameter names to their configuration
                       Format: {param_name: {'type': 'float', 'min': 0, 'max': 1, 'value': 0.5, 'step': 0.01}}

        Returns:
            Widget container with all controls
        """
        if not HAS_WIDGETS:
            raise ImportError("ipywidgets required for parameter controls")

        controls = []

        for param_name, config in parameters.items():
            param_type = config.get("type", "float")

            if param_type == "float":
                widget = widgets.FloatSlider(
                    value=config.get("value", 0.5),
                    min=config.get("min", 0.0),
                    max=config.get("max", 1.0),
                    step=config.get("step", 0.01),
                    description=param_name,
                    style={"description_width": "initial"},
                    layout=widgets.Layout(width=self.config.widget_layout_width),
                )
            elif param_type == "int":
                widget = widgets.IntSlider(
                    value=config.get("value", 10),
                    min=config.get("min", 1),
                    max=config.get("max", 100),
                    step=config.get("step", 1),
                    description=param_name,
                    style={"description_width": "initial"},
                    layout=widgets.Layout(width=self.config.widget_layout_width),
                )
            elif param_type == "bool":
                widget = widgets.Checkbox(
                    value=config.get("value", False),
                    description=param_name,
                    style={"description_width": "initial"},
                    layout=widgets.Layout(width=self.config.widget_layout_width),
                )
            elif param_type == "dropdown":
                widget = widgets.Dropdown(
                    options=config.get("options", ["Option 1", "Option 2"]),
                    value=config.get("value", config.get("options", ["Option 1"])[0]),
                    description=param_name,
                    style={"description_width": "initial"},
                    layout=widgets.Layout(width=self.config.widget_layout_width),
                )
            else:
                continue

            # Store widget
            self.controls[param_name] = widget

            # Add change handler
            widget.observe(self._on_parameter_change, names="value")

            controls.append(widget)

        # Create container
        container = widgets.VBox(controls)
        return container

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameter values."""
        return {name: widget.value for name, widget in self.controls.items()}

    def add_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Add a callback function that gets called when parameters change."""
        self.callbacks.append(callback)

    def _on_parameter_change(self, change):
        """Handle parameter changes."""
        for callback in self.callbacks:
            try:
                callback(self.get_parameters())
            except Exception as e:
                warnings.warn(f"Error in parameter callback: {e}", stacklevel=2)


class AnimatedPipelineFlow:
    """Animated visualization of data flowing through pipeline stages."""

    def __init__(self, config: JupyterVizConfig = None):
        self.config = config or JupyterVizConfig()
        self.animation_fig: FigureWidget | None = None
        self.is_animating = False
        self._animation_thread: threading.Thread | None = None

        if not HAS_PLOTLY:
            warnings.warn("Plotly not available. Install with: pip install plotly", stacklevel=2)

    def create_flow_animation(
        self, pipeline_stages: list[str], data_tracker: DataFlowTracker
    ) -> FigureWidget:
        """Create an animated visualization of data flow through pipeline stages.

        Args:
            pipeline_stages: List of pipeline stage names
            data_tracker: DataFlowTracker instance with recorded data

        Returns:
            Animated Plotly figure
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly required for flow animations")

        # Create figure with subplots for each stage
        fig = make_subplots(
            rows=len(pipeline_stages), cols=1, subplot_titles=pipeline_stages, vertical_spacing=0.1
        )

        # Add traces for each stage
        for i, stage in enumerate(pipeline_stages):
            fig.add_scatter(
                x=[],
                y=[],
                mode="lines+markers",
                name=f"{stage} Data",
                row=i + 1,
                col=1,
                line={"color": self.config.color_palette[i % len(self.config.color_palette)]},
            )

        # Configure layout
        fig.update_layout(
            title="Animated Pipeline Data Flow",
            height=self.config.plotly_height * len(pipeline_stages) // 2,
            width=self.config.plotly_width,
            template=self.config.plotly_theme,
            showlegend=True,
        )

        self.animation_fig = fig
        return fig

    def start_animation(self, data_tracker: DataFlowTracker) -> None:
        """Start the flow animation."""
        if not self.animation_fig:
            raise ValueError("Create flow animation first")

        self.is_animating = True
        self._animation_thread = threading.Thread(target=self._animate_flow, args=(data_tracker,))
        self._animation_thread.start()

    def stop_animation(self) -> None:
        """Stop the flow animation."""
        self.is_animating = False
        if self._animation_thread:
            self._animation_thread.join()

    def _animate_flow(self, data_tracker: DataFlowTracker) -> None:
        """Animation loop for data flow."""
        step_count = 0

        while self.is_animating:
            # Get recent data points
            recent_data = data_tracker.get_recent_data(50)

            if not recent_data:
                time.sleep(self.config.animation_interval_ms / 1000)
                continue

            # Group data by module
            module_data = defaultdict(list)
            for point in recent_data:
                module_data[point.module_name].append((point.step, point.value))

            # Update traces
            with self.animation_fig.batch_update():
                for _i, trace in enumerate(self.animation_fig.data):
                    module_name = trace.name.replace(" Data", "")
                    if module_name in module_data:
                        steps, values = zip(*module_data[module_name], strict=False)
                        trace.x = steps
                        trace.y = values

            step_count += 1
            time.sleep(self.config.animation_interval_ms / 1000)


def create_pipeline_dashboard(
    pipeline_fn: Any, input_example: Any, config: JupyterVizConfig = None
) -> dict[str, Any]:
    """Create a comprehensive interactive dashboard for a pipeline.

    Args:
        pipeline_fn: The streaming pipeline function
        input_example: Example input for the pipeline
        config: Visualization configuration

    Returns:
        Dictionary containing all dashboard components
    """
    if config is None:
        config = JupyterVizConfig()

    dashboard = {}

    # Create pipeline graph
    if HAS_PLOTLY and HAS_NETWORKX:
        graph_viz = InteractivePipelineGraph(config)
        dashboard["pipeline_graph"] = graph_viz.create_pipeline_graph(pipeline_fn, input_example)

    # Create streaming visualizer
    if HAS_PLOTLY:
        stream_viz = StreamingDataVisualizer(config)
        dashboard["streaming_plot"] = stream_viz.create_streaming_plot(
            title="Real-time Pipeline Data"
        )
        dashboard["stream_visualizer"] = stream_viz

    # Create parameter controls
    if HAS_WIDGETS:
        param_controls = InteractiveParameterControls(config)

        # Example parameter configuration - customize based on your pipeline
        example_params = {
            "learning_rate": {
                "type": "float",
                "min": 0.001,
                "max": 0.1,
                "value": 0.01,
                "step": 0.001,
            },
            "window_size": {"type": "int", "min": 5, "max": 100, "value": 20, "step": 5},
            "enable_adaptation": {"type": "bool", "value": True},
        }

        dashboard["parameter_panel"] = param_controls.create_parameter_panel(example_params)
        dashboard["parameter_controls"] = param_controls

    return dashboard


def display_pipeline_dashboard(dashboard: dict[str, Any]) -> None:
    """Display the complete pipeline dashboard in Jupyter.

    Args:
        dashboard: Dashboard components from create_pipeline_dashboard
    """
    if not HAS_WIDGETS:
        print("ipywidgets not available - displaying individual components")
        for _name, component in dashboard.items():
            if hasattr(component, "_ipython_display_"):
                display(component)
        return

    # Create layout
    components = []

    # Add title
    title = widgets.HTML("<h2>🎨 WAX-ML Pipeline Dashboard</h2>")
    components.append(title)

    # Add pipeline graph if available
    if "pipeline_graph" in dashboard:
        graph_title = widgets.HTML("<h3>📊 Pipeline Structure</h3>")
        components.append(graph_title)
        components.append(dashboard["pipeline_graph"])

    # Add parameter controls if available
    if "parameter_panel" in dashboard:
        param_title = widgets.HTML("<h3>🎛️ Interactive Controls</h3>")
        components.append(param_title)
        components.append(dashboard["parameter_panel"])

    # Add streaming plot if available
    if "streaming_plot" in dashboard:
        stream_title = widgets.HTML("<h3>📈 Real-time Data Stream</h3>")
        components.append(stream_title)
        components.append(dashboard["streaming_plot"])

    # Create and display the full dashboard
    full_dashboard = widgets.VBox(components)
    display(full_dashboard)


# Convenience functions for quick visualization
def quick_pipeline_viz(pipeline_fn: Any, input_example: Any) -> None:
    """Quick visualization of a pipeline with sensible defaults."""
    dashboard = create_pipeline_dashboard(pipeline_fn, input_example)
    display_pipeline_dashboard(dashboard)


def quick_streaming_plot(
    title: str = "Streaming Data",
) -> tuple[FigureWidget, StreamingDataVisualizer]:
    """Quick setup for streaming data visualization."""
    viz = StreamingDataVisualizer()
    fig = viz.create_streaming_plot(title=title)
    display(fig)
    return fig, viz
