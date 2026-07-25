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
"""Interactive dashboard for monitoring WAX-ML streaming pipelines.

This module provides a comprehensive web-based dashboard for monitoring
and analyzing streaming computation pipelines in real-time, including:
- Live pipeline visualization
- Performance monitoring
- Data flow tracking
- Interactive parameter tuning
- Historical analysis

Key features:
- Web-based interface accessible from any browser
- Real-time updates with WebSocket connections
- Interactive pipeline exploration
- Performance metrics and alerts
- Data export capabilities
- Multi-pipeline monitoring
- Custom dashboard layouts
"""

from __future__ import annotations

import threading
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

# Optional web framework dependencies
try:
    import flask
    from flask import Flask, jsonify, render_template, request
    from flask_socketio import SocketIO, emit

    HAS_FLASK = True
except ImportError:
    flask = None
    Flask = None
    render_template = None
    request = None
    jsonify = None
    SocketIO = None
    emit = None
    HAS_FLASK = False

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False

from .computation_graph import ComputationGraphRenderer
from .data_flow_visualizer import DataFlowTracker, DataFlowVisualizer


@dataclass
class DashboardConfig:
    """Configuration for the interactive dashboard."""

    # Server configuration
    host: str = "localhost"
    port: int = 8080
    debug: bool = False

    # Dashboard layout
    title: str = "WAX-ML Streaming Pipeline Dashboard"
    theme: str = "dark"  # 'dark', 'light'
    auto_refresh_interval: float = 1.0  # seconds

    # Data retention
    max_data_points: int = 1000
    max_history_hours: float = 24.0

    # Visualization settings
    default_plot_height: int = 400
    default_plot_width: int = 800
    show_grid: bool = True
    show_legend: bool = True

    # Monitoring settings
    enable_alerts: bool = True
    performance_threshold_ms: float = 100.0
    memory_threshold_mb: float = 1000.0

    # Export settings
    enable_export: bool = True
    export_formats: list[str] = field(default_factory=lambda: ["csv", "json", "png"])


@dataclass
class DashboardState:
    """Internal state of the dashboard."""

    active_pipelines: dict[str, Any] = field(default_factory=dict)
    data_trackers: dict[str, DataFlowTracker] = field(default_factory=dict)
    visualizers: dict[str, DataFlowVisualizer] = field(default_factory=dict)

    # Performance monitoring
    performance_metrics: dict[str, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=1000))
    )
    alerts: deque = field(default_factory=lambda: deque(maxlen=100))

    # User session data
    connected_clients: set = field(default_factory=set)
    last_update: float = field(default_factory=time.time)


class InteractiveDashboard:
    """Interactive web dashboard for WAX-ML streaming pipelines."""

    def __init__(self, config: DashboardConfig | None = None):
        """Initialize the interactive dashboard.

        Args:
            config: Dashboard configuration (uses defaults if None)
        """
        self.config = config or DashboardConfig()
        self.state = DashboardState()

        # Web framework setup
        self.app: Flask | None = None
        self.socketio: SocketIO | None = None
        self.server_thread: threading.Thread | None = None
        self.running = False

        # Initialize web framework
        if HAS_FLASK:
            self._setup_flask_app()
        elif HAS_STREAMLIT:
            self._setup_streamlit_app()
        else:
            warnings.warn(
                "No web framework available. Install flask or streamlit for dashboard.",
                stacklevel=2,
            )

    def _setup_flask_app(self):
        """Setup Flask application with routes and WebSocket."""
        self.app = Flask(__name__, template_folder=None)
        self.app.config["SECRET_KEY"] = "wax-ml-dashboard"

        if HAS_FLASK:
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")

            # Register routes
            self._register_flask_routes()
            self._register_socketio_events()

    def _register_flask_routes(self):
        """Register Flask HTTP routes."""

        @self.app.route("/")
        def dashboard_home():
            """Main dashboard page."""
            return self._render_dashboard_html()

        @self.app.route("/api/pipelines")
        def get_pipelines():
            """Get list of active pipelines."""
            return jsonify(
                {
                    "pipelines": list(self.state.active_pipelines.keys()),
                    "count": len(self.state.active_pipelines),
                }
            )

        @self.app.route("/api/pipeline/<pipeline_id>/data")
        def get_pipeline_data(pipeline_id):
            """Get data for a specific pipeline."""
            if pipeline_id not in self.state.data_trackers:
                return jsonify({"error": "Pipeline not found"}), 404

            tracker = self.state.data_trackers[pipeline_id]
            recent_data = tracker.get_recent_data(100)

            # Convert data to JSON-serializable format
            data_json = []
            for dp in recent_data:
                data_json.append(
                    {
                        "timestamp": dp.timestamp,
                        "step": dp.step,
                        "module_name": dp.module_name,
                        "data_type": dp.data_type,
                        "value": self._serialize_value(dp.value),
                        "shape": dp.shape,
                        "metadata": dp.metadata,
                    }
                )

            return jsonify(
                {"pipeline_id": pipeline_id, "data": data_json, "total_steps": tracker.step_count}
            )

        @self.app.route("/api/pipeline/<pipeline_id>/graph")
        def get_pipeline_graph(pipeline_id):
            """Get computation graph for a pipeline."""
            if pipeline_id not in self.state.active_pipelines:
                return jsonify({"error": "Pipeline not found"}), 404

            # Generate graph representation
            self.state.active_pipelines[pipeline_id]

            try:
                # Try to render graph using ComputationGraphRenderer
                ComputationGraphRenderer(output_format="json")
                # Note: This would need pipeline analysis capabilities
                graph_data = {
                    "nodes": [{"id": "main", "name": pipeline_id, "type": "pipeline"}],
                    "edges": [],
                }
            except Exception:
                graph_data = {
                    "nodes": [{"id": "main", "name": pipeline_id, "type": "pipeline"}],
                    "edges": [],
                }

            return jsonify(graph_data)

        @self.app.route("/api/performance")
        def get_performance_metrics():
            """Get overall performance metrics."""
            metrics = {}

            for metric_name, metric_data in self.state.performance_metrics.items():
                if metric_data:
                    recent_values = list(metric_data)[-50:]  # Last 50 points
                    metrics[metric_name] = {
                        "values": recent_values,
                        "current": recent_values[-1] if recent_values else 0,
                        "average": sum(recent_values) / len(recent_values) if recent_values else 0,
                    }

            return jsonify(metrics)

        @self.app.route("/api/alerts")
        def get_alerts():
            """Get recent alerts."""
            alerts = list(self.state.alerts)
            return jsonify({"alerts": alerts, "count": len(alerts)})

    def _register_socketio_events(self):
        """Register WebSocket events for real-time updates."""

        @self.socketio.on("connect")
        def handle_connect():
            """Handle client connection."""
            self.state.connected_clients.add(request.sid)
            emit("status", {"message": "Connected to WAX-ML Dashboard"})

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle client disconnection."""
            self.state.connected_clients.discard(request.sid)

        @self.socketio.on("subscribe_pipeline")
        def handle_subscribe(data):
            """Handle pipeline subscription."""
            pipeline_id = data.get("pipeline_id")
            if pipeline_id in self.state.active_pipelines:
                emit("pipeline_subscribed", {"pipeline_id": pipeline_id})
            else:
                emit("error", {"message": f"Pipeline {pipeline_id} not found"})

    def _setup_streamlit_app(self):
        """Setup Streamlit application."""
        # Streamlit setup would go here
        # Note: Streamlit has a different architecture, so this would be implemented differently
        pass

    def register_pipeline(
        self, pipeline_id: str, pipeline_fn: Callable, input_example: Any, description: str = ""
    ) -> None:
        """Register a streaming pipeline for monitoring.

        Args:
            pipeline_id: Unique identifier for the pipeline
            pipeline_fn: The streaming pipeline function
            input_example: Example input for analysis
            description: Optional description of the pipeline
        """
        # Store pipeline information
        self.state.active_pipelines[pipeline_id] = {
            "function": pipeline_fn,
            "input_example": input_example,
            "description": description,
            "registered_at": time.time(),
            "total_calls": 0,
        }

        # Create data tracker
        tracker = DataFlowTracker(
            max_history=self.config.max_data_points,
            track_inputs=True,
            track_outputs=True,
            track_states=True,
        )
        self.state.data_trackers[pipeline_id] = tracker

        # Create visualizer
        visualizer = DataFlowVisualizer(backend="plotly" if HAS_FLASK else "text")
        visualizer.attach_tracker(tracker)
        self.state.visualizers[pipeline_id] = visualizer

        print(f"✅ Registered pipeline: {pipeline_id}")

    def record_pipeline_data(
        self,
        pipeline_id: str,
        module_name: str,
        data_type: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record data from a pipeline execution.

        Args:
            pipeline_id: ID of the pipeline
            module_name: Name of the module producing data
            data_type: Type of data ('input', 'output', 'state', etc.)
            value: The actual data value
            metadata: Optional metadata
        """
        if pipeline_id not in self.state.data_trackers:
            warnings.warn(f"Pipeline {pipeline_id} not registered", stacklevel=2)
            return

        tracker = self.state.data_trackers[pipeline_id]
        tracker.record_data(module_name, data_type, value, metadata)

        # Update performance metrics
        self._update_performance_metrics(pipeline_id, metadata or {})

        # Broadcast update to connected clients
        if self.socketio and self.state.connected_clients:
            data_point = {
                "pipeline_id": pipeline_id,
                "module_name": module_name,
                "data_type": data_type,
                "value": self._serialize_value(value),
                "timestamp": time.time(),
                "step": tracker.step_count,
            }
            self.socketio.emit("pipeline_data_update", data_point)

    def step_pipeline(self, pipeline_id: str) -> None:
        """Increment step counter for a pipeline."""
        if pipeline_id in self.state.data_trackers:
            self.state.data_trackers[pipeline_id].step()

        if pipeline_id in self.state.active_pipelines:
            self.state.active_pipelines[pipeline_id]["total_calls"] += 1

    def _update_performance_metrics(self, pipeline_id: str, metadata: dict[str, Any]) -> None:
        """Update performance metrics from pipeline execution."""
        current_time = time.time()

        # Record execution time if available
        if "execution_time_ms" in metadata:
            exec_time = metadata["execution_time_ms"]
            self.state.performance_metrics[f"{pipeline_id}_execution_time"].append(exec_time)

            # Check for performance alerts
            if self.config.enable_alerts and exec_time > self.config.performance_threshold_ms:
                alert = {
                    "timestamp": current_time,
                    "type": "performance",
                    "severity": "warning",
                    "pipeline_id": pipeline_id,
                    "message": f"Slow execution: {exec_time:.1f}ms (threshold: {self.config.performance_threshold_ms}ms)",
                }
                self.state.alerts.append(alert)

                # Broadcast alert
                if self.socketio:
                    self.socketio.emit("alert", alert)

        # Record memory usage if available
        if "memory_usage_mb" in metadata:
            memory_mb = metadata["memory_usage_mb"]
            self.state.performance_metrics[f"{pipeline_id}_memory_usage"].append(memory_mb)

            # Check for memory alerts
            if self.config.enable_alerts and memory_mb > self.config.memory_threshold_mb:
                alert = {
                    "timestamp": current_time,
                    "type": "memory",
                    "severity": "warning",
                    "pipeline_id": pipeline_id,
                    "message": f"High memory usage: {memory_mb:.1f}MB (threshold: {self.config.memory_threshold_mb}MB)",
                }
                self.state.alerts.append(alert)

                if self.socketio:
                    self.socketio.emit("alert", alert)

        # Record throughput
        self.state.performance_metrics["global_throughput"].append(current_time)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON transmission."""
        if hasattr(value, "tolist"):  # NumPy/JAX arrays
            if value.size <= 10:  # Small arrays - include full data
                return value.tolist()
            else:  # Large arrays - include summary
                return {
                    "type": "array",
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "min": float(value.min()),
                    "max": float(value.max()),
                    "mean": float(value.mean()),
                }
        elif isinstance(value, int | float | str | bool):
            return value
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list | tuple):
            if len(value) <= 10:
                return [self._serialize_value(v) for v in value]
            else:
                return {"type": "sequence", "length": len(value)}
        else:
            return {"type": str(type(value).__name__), "str": str(value)}

    def _render_dashboard_html(self) -> str:
        """Render the main dashboard HTML page."""
        # This would normally use a proper template engine
        # For now, return a simple HTML page

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: {"#2b2b2b" if self.config.theme == "dark" else "#ffffff"};
            color: {"#ffffff" if self.config.theme == "dark" else "#000000"};
        }}
        .header {{
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .panel {{
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 20px;
            background-color: {"#3b3b3b" if self.config.theme == "dark" else "#f9f9f9"};
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 10px;
            border-radius: 5px;
            background-color: #4CAF50;
            color: white;
            min-width: 120px;
            text-align: center;
        }}
        .alert {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            background-color: #ff9800;
            color: white;
        }}
        .pipeline-list {{
            list-style: none;
            padding: 0;
        }}
        .pipeline-item {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            background-color: {"#4b4b4b" if self.config.theme == "dark" else "#e9e9e9"};
            cursor: pointer;
        }}
        .pipeline-item:hover {{
            background-color: {"#5b5b5b" if self.config.theme == "dark" else "#d9d9d9"};
        }}
        #status {{
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 10px;
            border-radius: 5px;
            background-color: #4CAF50;
            color: white;
        }}
    </style>
</head>
<body>
    <div id="status">Connecting...</div>

    <div class="header">
        <h1>{self.config.title}</h1>
        <div>
            <span class="metric">Pipelines: <span id="pipeline-count">0</span></span>
            <span class="metric">Active Clients: <span id="client-count">0</span></span>
            <span class="metric">Alerts: <span id="alert-count">0</span></span>
        </div>
    </div>

    <div class="grid">
        <div class="panel">
            <h2>Active Pipelines</h2>
            <ul class="pipeline-list" id="pipeline-list">
                <li>Loading...</li>
            </ul>
        </div>

        <div class="panel">
            <h2>Performance Metrics</h2>
            <div id="performance-plot" style="height: 300px;"></div>
        </div>

        <div class="panel">
            <h2>Data Flow</h2>
            <div id="dataflow-plot" style="height: 300px;"></div>
        </div>

        <div class="panel">
            <h2>Recent Alerts</h2>
            <div id="alerts-container">
                <p>No alerts</p>
            </div>
        </div>
    </div>

    <script>
        // Initialize WebSocket connection
        const socket = io();

        socket.on('connect', function() {{
            document.getElementById('status').textContent = 'Connected';
            document.getElementById('status').style.backgroundColor = '#4CAF50';
            loadDashboardData();
        }});

        socket.on('disconnect', function() {{
            document.getElementById('status').textContent = 'Disconnected';
            document.getElementById('status').style.backgroundColor = '#f44336';
        }});

        socket.on('pipeline_data_update', function(data) {{
            updateDataFlowPlot(data);
        }});

        socket.on('alert', function(alert) {{
            addAlert(alert);
        }});

        function loadDashboardData() {{
            // Load pipelines
            fetch('/api/pipelines')
                .then(response => response.json())
                .then(data => {{
                    updatePipelineList(data.pipelines);
                    document.getElementById('pipeline-count').textContent = data.count;
                }});

            // Load performance metrics
            fetch('/api/performance')
                .then(response => response.json())
                .then(data => updatePerformancePlot(data));

            // Load alerts
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {{
                    updateAlerts(data.alerts);
                    document.getElementById('alert-count').textContent = data.count;
                }});
        }}

        function updatePipelineList(pipelines) {{
            const list = document.getElementById('pipeline-list');
            list.innerHTML = '';

            pipelines.forEach(pipeline => {{
                const item = document.createElement('li');
                item.className = 'pipeline-item';
                item.textContent = pipeline;
                item.onclick = () => selectPipeline(pipeline);
                list.appendChild(item);
            }});

            if (pipelines.length === 0) {{
                const item = document.createElement('li');
                item.textContent = 'No active pipelines';
                list.appendChild(item);
            }}
        }}

        function updatePerformancePlot(data) {{
            const traces = [];

            Object.keys(data).forEach(metric => {{
                if (data[metric].values) {{
                    traces.push({{
                        y: data[metric].values,
                        type: 'scatter',
                        mode: 'lines',
                        name: metric
                    }});
                }}
            }});

            const layout = {{
                title: 'Performance Metrics',
                xaxis: {{ title: 'Time' }},
                yaxis: {{ title: 'Value' }},
                paper_bgcolor: '{"#3b3b3b" if self.config.theme == "dark" else "#ffffff"}',
                plot_bgcolor: '{"#2b2b2b" if self.config.theme == "dark" else "#f9f9f9"}',
                font: {{ color: '{"#ffffff" if self.config.theme == "dark" else "#000000"}' }}
            }};

            Plotly.newPlot('performance-plot', traces, layout);
        }}

        function updateDataFlowPlot(data) {{
            // Update data flow visualization
            console.log('Data flow update:', data);
        }}

        function addAlert(alert) {{
            const container = document.getElementById('alerts-container');
            const alertDiv = document.createElement('div');
            alertDiv.className = 'alert';
            alertDiv.innerHTML = `
                <strong>${{alert.type.toUpperCase()}}</strong> - ${{alert.pipeline_id}}<br>
                ${{alert.message}}<br>
                <small>${{new Date(alert.timestamp * 1000).toLocaleString()}}</small>
            `;
            container.insertBefore(alertDiv, container.firstChild);

            // Keep only last 10 alerts visible
            while (container.children.length > 10) {{
                container.removeChild(container.lastChild);
            }}

            // Update alert count
            const alertCount = document.getElementById('alert-count');
            alertCount.textContent = parseInt(alertCount.textContent) + 1;
        }}

        function updateAlerts(alerts) {{
            const container = document.getElementById('alerts-container');
            container.innerHTML = '';

            if (alerts.length === 0) {{
                container.innerHTML = '<p>No alerts</p>';
                return;
            }}

            alerts.slice(-10).forEach(alert => {{
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert';
                alertDiv.innerHTML = `
                    <strong>${{alert.type.toUpperCase()}}</strong> - ${{alert.pipeline_id}}<br>
                    ${{alert.message}}<br>
                    <small>${{new Date(alert.timestamp * 1000).toLocaleString()}}</small>
                `;
                container.appendChild(alertDiv);
            }});
        }}

        function selectPipeline(pipelineId) {{
            console.log('Selected pipeline:', pipelineId);
            socket.emit('subscribe_pipeline', {{ pipeline_id: pipelineId }});
        }}

        // Auto-refresh every {self.config.auto_refresh_interval} seconds
        setInterval(loadDashboardData, {self.config.auto_refresh_interval * 1000});
    </script>
</body>
</html>
"""
        return html

    def start_server(self, blocking: bool = True) -> None:
        """Start the dashboard server.

        Args:
            blocking: Whether to block the current thread
        """
        if not HAS_FLASK:
            print("❌ Flask not available. Install with: pip install flask flask-socketio")
            return

        if self.running:
            print("⚠️ Dashboard server already running")
            return

        print(f"🚀 Starting WAX-ML Dashboard on http://{self.config.host}:{self.config.port}")

        def run_server():
            self.running = True
            try:
                self.socketio.run(
                    self.app,
                    host=self.config.host,
                    port=self.config.port,
                    debug=self.config.debug,
                    allow_unsafe_werkzeug=True,
                )
            except Exception as e:
                print(f"❌ Dashboard server error: {e}")
            finally:
                self.running = False

        if blocking:
            run_server()
        else:
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            time.sleep(1)  # Give server time to start

    def stop_server(self) -> None:
        """Stop the dashboard server."""
        if self.running and self.socketio:
            print("🛑 Stopping WAX-ML Dashboard")
            self.running = False
            # Note: SocketIO doesn't have a clean shutdown method
            # In practice, you'd need to implement proper shutdown handling


# Convenience functions


def create_pipeline_dashboard(
    pipelines: Mapping[str, tuple[Callable, Any]],
    config: DashboardConfig | None = None,
    start_server: bool = True,
) -> InteractiveDashboard:
    """Create and optionally start a pipeline monitoring dashboard.

    Args:
        pipelines: Mapping of pipeline_id -> (function, input_example)
        config: Dashboard configuration
        start_server: Whether to start the server immediately

    Returns:
        Configured dashboard instance
    """
    dashboard = InteractiveDashboard(config)

    # Register all pipelines
    for pipeline_id, (pipeline_fn, input_example) in pipelines.items():
        dashboard.register_pipeline(pipeline_id, pipeline_fn, input_example)

    if start_server:
        dashboard.start_server(blocking=False)

    return dashboard


def launch_monitoring_server(
    host: str = "localhost", port: int = 8080, **kwargs
) -> InteractiveDashboard:
    """Launch a monitoring server for WAX-ML pipelines.

    Args:
        host: Server host address
        port: Server port number
        **kwargs: Additional configuration options

    Returns:
        Running dashboard instance
    """
    config = DashboardConfig(host=host, port=port, **kwargs)
    dashboard = InteractiveDashboard(config)

    print(f"🌐 WAX-ML Dashboard available at: http://{host}:{port}")
    print("📊 Register pipelines using dashboard.register_pipeline()")
    print("🔄 Record data using dashboard.record_pipeline_data()")

    return dashboard
