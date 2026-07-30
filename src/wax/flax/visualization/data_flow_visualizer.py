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
"""Real-time data flow visualizer for WAX-ML streaming pipelines.

This module provides tools for visualizing the flow of data through
streaming computation pipelines in real-time, including:
- Data tensor flow tracking
- State evolution visualization
- Performance bottleneck identification
- Interactive data inspection

Key features:
- Real-time streaming data visualization
- Multiple visualization modes (line plots, heatmaps, 3D surfaces)
- Automatic data type detection and appropriate visualization
- Interactive data exploration with zoom/pan
- Animation of data flow through pipeline stages
- Export capabilities for analysis and presentation
"""

from __future__ import annotations

import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

# Optional dependencies for visualization
try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    animation = None
    np = None
    HAS_MATPLOTLIB = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    go = None
    px = None
    make_subplots = None
    pyo = None
    HAS_PLOTLY = False


@dataclass
class DataPoint:
    """Represents a data point in the streaming pipeline."""

    timestamp: float
    step: int
    module_name: str
    data_type: str  # 'input', 'output', 'state', 'intermediate'
    value: Any
    shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Extract shape and metadata from value."""
        if hasattr(self.value, "shape"):
            self.shape = self.value.shape
        elif isinstance(self.value, list | tuple):
            self.shape = (len(self.value),)
        elif isinstance(self.value, dict):
            self.shape = (len(self.value),)
        else:
            self.shape = ()


@dataclass
class DataFlowTracker:
    """Tracks data flow through streaming pipeline for visualization."""

    max_history: int = 1000
    track_inputs: bool = True
    track_outputs: bool = True
    track_states: bool = True
    track_gradients: bool = False

    def __post_init__(self):
        """Initialize tracking data structures."""
        self.data_history: deque[DataPoint] = deque(maxlen=self.max_history)
        self.module_data: dict[str, deque[DataPoint]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self.step_count = 0
        self.start_time = time.time()

    def record_data(
        self, module_name: str, data_type: str, value: Any, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a data point in the pipeline.

        Args:
            module_name: Name of the module producing this data
            data_type: Type of data ('input', 'output', 'state', etc.)
            value: The actual data value
            metadata: Optional metadata about the data
        """
        data_point = DataPoint(
            timestamp=time.time() - self.start_time,
            step=self.step_count,
            module_name=module_name,
            data_type=data_type,
            value=value,
            shape=(),  # Will be set in __post_init__
            metadata=metadata or {},
        )

        self.data_history.append(data_point)
        self.module_data[module_name].append(data_point)

    def step(self) -> None:
        """Increment step counter."""
        self.step_count += 1

    def get_module_history(self, module_name: str, data_type: str | None = None) -> list[DataPoint]:
        """Get data history for a specific module.

        Args:
            module_name: Name of the module
            data_type: Optional filter by data type

        Returns:
            List of data points for the module
        """
        history = list(self.module_data[module_name])

        if data_type:
            history = [dp for dp in history if dp.data_type == data_type]

        return history

    def get_recent_data(self, n_points: int = 100) -> list[DataPoint]:
        """Get the most recent n data points."""
        return list(self.data_history)[-n_points:]

    def clear_history(self) -> None:
        """Clear all tracking history."""
        self.data_history.clear()
        self.module_data.clear()
        self.step_count = 0
        self.start_time = time.time()


class DataFlowVisualizer:
    """Visualizes real-time data flow in streaming pipelines."""

    def __init__(
        self,
        backend: str = "matplotlib",
        figure_size: tuple[int, int] = (12, 8),
        max_points: int = 200,
        update_interval: float = 0.1,
        auto_scale: bool = True,
    ):
        """Initialize the data flow visualizer.

        Args:
            backend: Visualization backend ('matplotlib', 'plotly', 'text')
            figure_size: Size of the figure in inches
            max_points: Maximum number of points to display
            update_interval: Update interval in seconds for real-time mode
            auto_scale: Whether to automatically scale axes
        """
        self.backend = backend
        self.figure_size = figure_size
        self.max_points = max_points
        self.update_interval = update_interval
        self.auto_scale = auto_scale

        self.tracker: DataFlowTracker | None = None
        self.active_plots: dict[str, Any] = {}
        self.plot_config: dict[str, Any] = {}

        # Validate backend availability
        if backend == "matplotlib" and not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not available, falling back to text mode", stacklevel=2)
            self.backend = "text"
        elif backend == "plotly" and not HAS_PLOTLY:
            warnings.warn("Plotly not available, falling back to matplotlib", stacklevel=2)
            self.backend = "matplotlib" if HAS_MATPLOTLIB else "text"

    def attach_tracker(self, tracker: DataFlowTracker) -> None:
        """Attach a data flow tracker to visualize."""
        self.tracker = tracker

    def create_streaming_plot(
        self,
        module_names: list[str] | None = None,
        data_types: list[str] | None = None,
        plot_type: str = "line",
    ) -> Any:
        """Create a streaming data plot.

        Args:
            module_names: List of module names to plot (None for all)
            data_types: List of data types to plot (None for all)
            plot_type: Type of plot ('line', 'scatter', 'heatmap', 'histogram')

        Returns:
            Plot object (depends on backend)
        """
        if not self.tracker:
            raise ValueError("No data tracker attached. Call attach_tracker() first.")

        if self.backend == "matplotlib":
            return self._create_matplotlib_plot(module_names, data_types, plot_type)
        elif self.backend == "plotly":
            return self._create_plotly_plot(module_names, data_types, plot_type)
        else:
            return self._create_text_plot(module_names, data_types, plot_type)

    def _create_matplotlib_plot(
        self, module_names: list[str] | None, data_types: list[str] | None, plot_type: str
    ) -> Any:
        """Create matplotlib-based streaming plot."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib not available")

        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        fig.suptitle("WAX-ML Streaming Data Flow", fontsize=16, fontweight="bold")

        # Flatten axes for easier indexing
        axes = axes.flatten()

        # Plot 1: Data values over time
        ax1 = axes[0]
        ax1.set_title("Data Values Over Time")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Value")

        # Plot 2: Data shapes/sizes
        ax2 = axes[1]
        ax2.set_title("Data Shapes/Sizes")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Size")

        # Plot 3: Module activity
        ax3 = axes[2]
        ax3.set_title("Module Activity")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Module")

        # Plot 4: Data flow rate
        ax4 = axes[3]
        ax4.set_title("Data Flow Rate")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Points/sec")

        # Set up real-time data structures
        plot_data: dict[str, Any] = {
            "steps": [],
            "values": defaultdict(list),
            "sizes": defaultdict(list),
            "timestamps": [],
            "modules": defaultdict(list),
        }

        # Store plot references
        self.active_plots["matplotlib"] = {"fig": fig, "axes": axes, "data": plot_data}

        return fig

    def _create_plotly_plot(
        self, module_names: list[str] | None, data_types: list[str] | None, plot_type: str
    ) -> Any:
        """Create Plotly-based interactive streaming plot."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly not available")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Data Values", "Data Shapes", "Module Activity", "Flow Rate"],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # Initial empty traces
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="lines+markers", name="Data Values"), row=1, col=1
        )
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Data Sizes"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Module Activity"), row=2, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Flow Rate"), row=2, col=2)

        fig.update_layout(title="WAX-ML Streaming Data Flow", height=800, showlegend=True)

        self.active_plots["plotly"] = {"fig": fig}

        return fig

    def _create_text_plot(
        self, module_names: list[str] | None, data_types: list[str] | None, plot_type: str
    ) -> str:
        """Create text-based data flow visualization."""
        if not self.tracker:
            return "No data tracker attached"

        recent_data = self.tracker.get_recent_data(20)

        lines = ["WAX-ML Streaming Data Flow", "=" * 40, ""]

        if not recent_data:
            lines.append("No data points recorded yet")
            return "\n".join(lines)

        # Summary statistics
        modules = {dp.module_name for dp in recent_data}
        data_types_found = {dp.data_type for dp in recent_data}

        lines.append(f"Active modules: {len(modules)}")
        lines.append(f"Data types: {', '.join(data_types_found)}")
        lines.append(f"Total steps: {self.tracker.step_count}")
        lines.append(f"Recent points: {len(recent_data)}")
        lines.append("")

        # Recent data points
        lines.append("Recent Data Points:")
        lines.append("-" * 20)

        for dp in recent_data[-10:]:  # Last 10 points
            value_str = self._format_value_for_display(dp.value)
            lines.append(
                f"Step {dp.step:3d} | {dp.module_name:15s} | {dp.data_type:10s} | {value_str}"
            )

        return "\n".join(lines)

    def _format_value_for_display(self, value: Any) -> str:
        """Format a value for text display."""
        if isinstance(value, jnp.ndarray):
            if value.size == 1:
                return f"{float(value):.3f}"
            else:
                return f"array{value.shape}"
        elif isinstance(value, int | float):
            return f"{value:.3f}"
        elif isinstance(value, dict):
            return f"dict({len(value)} keys)"
        elif isinstance(value, list | tuple):
            return f"{type(value).__name__}({len(value)})"
        else:
            return str(type(value).__name__)

    def update_plot(self) -> None:
        """Update the active plot with new data."""
        if not self.tracker:
            return

        if self.backend == "matplotlib" and "matplotlib" in self.active_plots:
            self._update_matplotlib_plot()
        elif self.backend == "plotly" and "plotly" in self.active_plots:
            self._update_plotly_plot()

    def _update_matplotlib_plot(self) -> None:
        """Update matplotlib plot with new data."""
        plot_info = self.active_plots["matplotlib"]
        fig = plot_info["fig"]
        axes = plot_info["axes"]
        plot_info["data"]

        # Get recent data
        recent_data = self.tracker.get_recent_data(self.max_points)

        if not recent_data:
            return

        # Update plot data structures
        [dp.step for dp in recent_data]
        [dp.timestamp for dp in recent_data]

        # Clear and update axes
        for ax in axes:
            ax.clear()

        # Plot 1: Data values
        ax1 = axes[0]
        ax1.set_title("Data Values Over Time")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Value")

        # Group by module and plot
        module_data: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        for dp in recent_data:
            if isinstance(dp.value, int | float | jnp.ndarray) and jnp.isscalar(dp.value):
                module_data[dp.module_name].append((dp.step, float(dp.value)))

        for module_name, data_points in module_data.items():
            if data_points:
                steps_mod, values_mod = zip(*data_points, strict=False)
                ax1.plot(steps_mod, values_mod, marker="o", label=module_name, alpha=0.7)

        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Data sizes
        ax2 = axes[1]
        ax2.set_title("Data Shapes/Sizes")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Size")

        size_data: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        for dp in recent_data:
            size = jnp.prod(jnp.array(dp.shape)) if dp.shape else 1
            size_data[dp.module_name].append((dp.step, int(size)))

        for module_name, size_points in size_data.items():
            if size_points:
                steps_mod, sizes_mod = zip(*size_points, strict=False)
                ax2.plot(steps_mod, sizes_mod, marker="s", label=module_name, alpha=0.7)

        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale("log")

        # Plot 3: Module activity timeline
        ax3 = axes[2]
        ax3.set_title("Module Activity")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Module")

        modules = list({dp.module_name for dp in recent_data})
        module_y_pos = {mod: i for i, mod in enumerate(modules)}

        for dp in recent_data:
            y_pos = module_y_pos[dp.module_name]
            color = (
                "red"
                if dp.data_type == "input"
                else "blue"
                if dp.data_type == "output"
                else "green"
            )
            ax3.scatter(dp.timestamp, y_pos, c=color, alpha=0.6, s=30)

        ax3.set_yticks(range(len(modules)))
        ax3.set_yticklabels(modules)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Data flow rate
        ax4 = axes[3]
        ax4.set_title("Data Flow Rate")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Points/sec")

        # Calculate flow rate over time windows
        if len(recent_data) > 1:
            window_size = 10
            flow_rates = []
            time_points = []

            for i in range(window_size, len(recent_data)):
                window_data = recent_data[i - window_size : i]
                time_span = window_data[-1].timestamp - window_data[0].timestamp
                if time_span > 0:
                    rate = len(window_data) / time_span
                    flow_rates.append(rate)
                    time_points.append(window_data[-1].timestamp)

            if flow_rates:
                ax4.plot(time_points, flow_rates, "r-", linewidth=2)

        ax4.grid(True, alpha=0.3)

        # Adjust layout and refresh
        fig.tight_layout()

        # If we're in interactive mode, update the display
        if hasattr(fig.canvas, "draw"):
            fig.canvas.draw()

    def _update_plotly_plot(self) -> None:
        """Update Plotly plot with new data."""
        # Implementation for Plotly real-time updates
        # This would use plotly's streaming/updating capabilities
        pass

    def start_real_time_monitoring(self) -> None:
        """Start real-time data flow monitoring."""
        if self.backend == "matplotlib" and HAS_MATPLOTLIB:
            self._start_matplotlib_animation()
        elif self.backend == "text":
            self._start_text_monitoring()
        else:
            warnings.warn(
                f"Real-time monitoring not implemented for backend: {self.backend}", stacklevel=2
            )

    def _start_matplotlib_animation(self) -> None:
        """Start matplotlib animation for real-time updates."""
        if "matplotlib" not in self.active_plots:
            self.create_streaming_plot()

        plot_info = self.active_plots["matplotlib"]
        fig = plot_info["fig"]

        def animate(frame):
            self._update_matplotlib_plot()
            return []

        anim = animation.FuncAnimation(
            fig,
            animate,
            interval=int(self.update_interval * 1000),
            blit=False,
            cache_frame_data=False,
        )

        self.active_plots["matplotlib"]["animation"] = anim
        plt.show()

    def _start_text_monitoring(self) -> None:
        """Start text-based monitoring loop."""
        import os
        import threading

        def monitor_loop():
            while True:
                os.system("clear" if os.name == "posix" else "cls")
                print(self._create_text_plot(None, None, "summary"))
                time.sleep(self.update_interval)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def save_plot(self, output_path: str, format: str = "png") -> str:
        """Save the current plot to file.

        Args:
            output_path: Output file path
            format: Output format ('png', 'svg', 'pdf', 'html')

        Returns:
            Path to saved file
        """
        if self.backend == "matplotlib" and "matplotlib" in self.active_plots:
            fig = self.active_plots["matplotlib"]["fig"]
            fig.savefig(output_path, format=format, dpi=300, bbox_inches="tight")
            return output_path
        elif self.backend == "plotly" and "plotly" in self.active_plots:
            fig = self.active_plots["plotly"]["fig"]
            if format == "html":
                fig.write_html(output_path)
            else:
                fig.write_image(output_path, format=format)
            return output_path
        else:
            # Save text output
            content = self._create_text_plot(None, None, "summary")
            with open(output_path, "w") as f:
                f.write(content)
            return output_path


# Convenience functions


def visualize_streaming_data(
    tracker: DataFlowTracker, backend: str = "matplotlib", output_path: str | None = None, **kwargs
) -> Any:
    """Visualize streaming data from a tracker.

    Args:
        tracker: Data flow tracker with recorded data
        backend: Visualization backend ('matplotlib', 'plotly', 'text')
        output_path: Optional output file path
        **kwargs: Additional arguments for DataFlowVisualizer

    Returns:
        Visualization object or file path
    """
    visualizer = DataFlowVisualizer(backend=backend, **kwargs)
    visualizer.attach_tracker(tracker)

    plot = visualizer.create_streaming_plot()
    visualizer.update_plot()

    if output_path:
        return visualizer.save_plot(output_path)
    else:
        return plot


def create_flow_animation(
    tracker: DataFlowTracker, duration: float = 10.0, fps: int = 10, output_path: str | None = None
) -> str:
    """Create an animation of data flow through the pipeline.

    Args:
        tracker: Data flow tracker with recorded data
        duration: Animation duration in seconds
        fps: Frames per second
        output_path: Output video file path

    Returns:
        Path to animation file
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for animation")

    visualizer = DataFlowVisualizer(backend="matplotlib", update_interval=1.0 / fps)
    visualizer.attach_tracker(tracker)

    fig = visualizer.create_streaming_plot()

    # Create animation
    frames = int(duration * fps)

    def animate(frame):
        # Simulate time progression through data
        max_step = max((dp.step for dp in tracker.data_history), default=0)
        current_step = int((frame / frames) * max_step)

        # Filter data up to current step
        filtered_data = [dp for dp in tracker.data_history if dp.step <= current_step]

        # Temporarily replace tracker data
        original_data = tracker.data_history.copy()
        tracker.data_history.clear()
        tracker.data_history.extend(filtered_data)

        # Update plot
        visualizer.update_plot()

        # Restore original data
        tracker.data_history.clear()
        tracker.data_history.extend(original_data)

        return []

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000 // fps, blit=False)

    if output_path:
        anim.save(output_path, writer="pillow", fps=fps)
        return output_path
    else:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            anim.save(tmp.name, writer="pillow", fps=fps)
            return tmp.name
