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
"""Tests for WAX-ML visualization tools."""

import tempfile
from unittest.mock import patch

import jax
import jax.numpy as jnp

from wax.flax.core.streaming_transforms import streaming_transform_with_state
from wax.flax.modules.buffer import Buffer
from wax.flax.modules.ewma import EWMA
from wax.flax.visualization import (
    ComputationGraphRenderer,
    DashboardConfig,
    DataFlowTracker,
    DataFlowVisualizer,
    InteractiveDashboard,
    PipelineEdge,
    PipelineNode,
    render_pipeline_graph,
    visualize_streaming_data,
)


class TestComputationGraphRenderer:
    """Test computation graph rendering functionality."""

    def test_basic_graph_renderer_creation(self):
        """Test basic renderer initialization."""
        renderer = ComputationGraphRenderer()

        assert renderer.output_format == "png"
        assert renderer.include_shapes
        assert not renderer.include_parameters
        assert len(renderer.nodes) == 0
        assert len(renderer.edges) == 0

    def test_add_nodes_and_edges(self):
        """Test adding nodes and edges to graph."""
        renderer = ComputationGraphRenderer()

        # Add a node
        node = PipelineNode(
            id="test_node",
            name="Test Node",
            module_type="EWMA",
            input_shapes={"input": (10,)},
            parameters={"alpha": 0.1},
        )
        renderer.add_node(node)

        assert "test_node" in renderer.nodes
        assert renderer.nodes["test_node"].name == "Test Node"
        assert renderer.nodes["test_node"].module_type == "EWMA"

        # Add an edge
        edge = PipelineEdge(source="input", target="test_node", label="data")
        renderer.add_edge(edge)

        assert len(renderer.edges) == 1
        assert renderer.edges[0].source == "input"
        assert renderer.edges[0].target == "test_node"

    def test_analyze_streaming_function(self):
        """Test analyzing a streaming function structure."""

        @streaming_transform_with_state
        def simple_ewma(x):
            return EWMA(alpha=0.1)(x)

        renderer = ComputationGraphRenderer()
        rng = jax.random.PRNGKey(42)

        # This should not raise an exception
        renderer.analyze_streaming_function(simple_ewma, jnp.array(1.0), rng)

        # Should have at least one node
        assert len(renderer.nodes) >= 1

    def test_render_to_text(self):
        """Test rendering graph to text format."""
        renderer = ComputationGraphRenderer(output_format="text")

        # Add simple graph
        node = PipelineNode(id="main", name="Main", module_type="Function")
        renderer.add_node(node)

        text_output = renderer.render()

        assert "WAX-ML Streaming Pipeline Graph" in text_output
        assert "main: Main (Function)" in text_output

    def test_render_to_dot(self):
        """Test rendering graph to DOT format."""
        renderer = ComputationGraphRenderer(output_format="dot")

        # Add simple graph
        node = PipelineNode(id="main", name="Main", module_type="Function")
        renderer.add_node(node)

        dot_output = renderer.render()

        assert "digraph WAX_ML_Pipeline" in dot_output
        assert '"main"' in dot_output

    def test_render_to_html(self):
        """Test rendering graph to HTML format."""
        renderer = ComputationGraphRenderer(output_format="html")

        # Add simple graph
        node = PipelineNode(id="main", name="Main", module_type="Function")
        renderer.add_node(node)

        html_output = renderer.render()

        assert "<!DOCTYPE html>" in html_output
        assert "WAX-ML Pipeline Visualization" in html_output
        assert "Main" in html_output

    def test_convenience_function(self):
        """Test render_pipeline_graph convenience function."""

        @streaming_transform_with_state
        def simple_function(x):
            return EWMA(alpha=0.2)(x)

        # Should not raise exception
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            output_path = render_pipeline_graph(
                simple_function, jnp.array(1.0), output_path=tmp.name, format="text"
            )

            assert output_path == tmp.name

            # Check file was created
            with open(tmp.name) as f:
                content = f.read()
                assert "WAX-ML Streaming Pipeline Graph" in content


class TestDataFlowTracker:
    """Test data flow tracking functionality."""

    def test_data_flow_tracker_creation(self):
        """Test basic tracker initialization."""
        tracker = DataFlowTracker()

        assert tracker.max_history == 1000
        assert tracker.track_inputs
        assert tracker.track_outputs
        assert tracker.step_count == 0
        assert len(tracker.data_history) == 0

    def test_record_data(self):
        """Test recording data points."""
        tracker = DataFlowTracker()

        # Record some data
        tracker.record_data("module1", "input", jnp.array([1.0, 2.0]), {"test": True})
        tracker.record_data("module1", "output", jnp.array([1.1, 2.1]))

        assert len(tracker.data_history) == 2

        data_point = tracker.data_history[0]
        assert data_point.module_name == "module1"
        assert data_point.data_type == "input"
        assert data_point.shape == (2,)
        assert data_point.metadata["test"]

    def test_step_counter(self):
        """Test step counter functionality."""
        tracker = DataFlowTracker()

        initial_step = tracker.step_count
        tracker.step()

        assert tracker.step_count == initial_step + 1

    def test_get_module_history(self):
        """Test retrieving module-specific history."""
        tracker = DataFlowTracker()

        # Record data for different modules
        tracker.record_data("module1", "input", 1.0)
        tracker.record_data("module2", "input", 2.0)
        tracker.record_data("module1", "output", 1.5)

        module1_history = tracker.get_module_history("module1")
        module2_history = tracker.get_module_history("module2")

        assert len(module1_history) == 2
        assert len(module2_history) == 1

        # Test filtering by data type
        module1_inputs = tracker.get_module_history("module1", "input")
        assert len(module1_inputs) == 1
        assert module1_inputs[0].data_type == "input"

    def test_get_recent_data(self):
        """Test retrieving recent data points."""
        tracker = DataFlowTracker()

        # Record multiple data points
        for i in range(20):
            tracker.record_data(f"module{i}", "input", float(i))

        recent_data = tracker.get_recent_data(10)

        assert len(recent_data) == 10
        # Should be the last 10 points
        assert recent_data[-1].value == 19.0

    def test_clear_history(self):
        """Test clearing tracking history."""
        tracker = DataFlowTracker()

        # Record some data
        tracker.record_data("module1", "input", 1.0)
        tracker.step()

        assert len(tracker.data_history) > 0
        assert tracker.step_count > 0

        tracker.clear_history()

        assert len(tracker.data_history) == 0
        assert tracker.step_count == 0


class TestDataFlowVisualizer:
    """Test data flow visualization functionality."""

    def test_data_flow_visualizer_creation(self):
        """Test basic visualizer initialization."""
        visualizer = DataFlowVisualizer(backend="text")

        assert visualizer.backend == "text"
        assert visualizer.max_points == 200
        assert visualizer.tracker is None

    def test_attach_tracker(self):
        """Test attaching a data tracker."""
        visualizer = DataFlowVisualizer(backend="text")
        tracker = DataFlowTracker()

        visualizer.attach_tracker(tracker)

        assert visualizer.tracker is tracker

    def test_create_text_plot(self):
        """Test creating text-based visualization."""
        visualizer = DataFlowVisualizer(backend="text")
        tracker = DataFlowTracker()

        # Record some test data
        tracker.record_data("module1", "input", 1.0)
        tracker.record_data("module1", "output", 1.1)
        tracker.step()

        visualizer.attach_tracker(tracker)

        plot_output = visualizer.create_streaming_plot()

        assert "WAX-ML Streaming Data Flow" in plot_output
        assert "module1" in plot_output
        assert "Total steps: 1" in plot_output

    def test_update_plot(self):
        """Test updating plot with new data."""
        visualizer = DataFlowVisualizer(backend="text")
        tracker = DataFlowTracker()

        visualizer.attach_tracker(tracker)

        # Should not raise exception even with no data
        visualizer.update_plot()

        # Add some data and update again
        tracker.record_data("module1", "input", 1.0)
        visualizer.update_plot()

    def test_save_plot_text(self):
        """Test saving text plot to file."""
        visualizer = DataFlowVisualizer(backend="text")
        tracker = DataFlowTracker()

        tracker.record_data("module1", "input", 1.0)
        visualizer.attach_tracker(tracker)
        visualizer.create_streaming_plot()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            output_path = visualizer.save_plot(tmp.name)

            assert output_path == tmp.name

            # Check file content
            with open(tmp.name) as f:
                content = f.read()
                assert "WAX-ML Streaming Data Flow" in content

    def test_convenience_function(self):
        """Test visualize_streaming_data convenience function."""
        tracker = DataFlowTracker()

        # Record some data
        tracker.record_data("module1", "input", 1.0)
        tracker.record_data("module1", "output", 1.1)

        # Should not raise exception
        plot = visualize_streaming_data(tracker, backend="text")

        assert "WAX-ML Streaming Data Flow" in plot


class TestInteractiveDashboard:
    """Test interactive dashboard functionality."""

    def test_dashboard_config(self):
        """Test dashboard configuration."""
        config = DashboardConfig(host="0.0.0.0", port=9090, title="Test Dashboard")

        assert config.host == "0.0.0.0"
        assert config.port == 9090
        assert config.title == "Test Dashboard"
        assert config.max_data_points == 1000

    def test_dashboard_creation(self):
        """Test basic dashboard initialization."""
        config = DashboardConfig(port=8081)  # Use different port
        dashboard = InteractiveDashboard(config)

        assert dashboard.config.port == 8081
        assert len(dashboard.state.active_pipelines) == 0
        assert len(dashboard.state.data_trackers) == 0

    def test_register_pipeline(self):
        """Test registering a pipeline for monitoring."""
        dashboard = InteractiveDashboard()

        @streaming_transform_with_state
        def test_pipeline(x):
            return EWMA(alpha=0.1)(x)

        dashboard.register_pipeline(
            "test_pipeline", test_pipeline, jnp.array(1.0), "Test pipeline description"
        )

        assert "test_pipeline" in dashboard.state.active_pipelines
        assert "test_pipeline" in dashboard.state.data_trackers
        assert "test_pipeline" in dashboard.state.visualizers

        pipeline_info = dashboard.state.active_pipelines["test_pipeline"]
        assert pipeline_info["description"] == "Test pipeline description"
        assert pipeline_info["total_calls"] == 0

    def test_record_pipeline_data(self):
        """Test recording pipeline execution data."""
        dashboard = InteractiveDashboard()

        # Register a pipeline first
        @streaming_transform_with_state
        def test_pipeline(x):
            return Buffer(maxlen=5)(x)

        dashboard.register_pipeline("test_pipeline", test_pipeline, jnp.array(1.0))

        # Record some data
        dashboard.record_pipeline_data(
            "test_pipeline",
            "buffer_module",
            "input",
            jnp.array([1.0, 2.0]),
            {"execution_time_ms": 5.0},
        )

        tracker = dashboard.state.data_trackers["test_pipeline"]
        assert len(tracker.data_history) == 1

        data_point = tracker.data_history[0]
        assert data_point.module_name == "buffer_module"
        assert data_point.data_type == "input"

    def test_step_pipeline(self):
        """Test stepping a registered pipeline."""
        dashboard = InteractiveDashboard()

        @streaming_transform_with_state
        def test_pipeline(x):
            return EWMA(alpha=0.1)(x)

        dashboard.register_pipeline("test_pipeline", test_pipeline, jnp.array(1.0))

        initial_steps = dashboard.state.data_trackers["test_pipeline"].step_count
        initial_calls = dashboard.state.active_pipelines["test_pipeline"]["total_calls"]

        dashboard.step_pipeline("test_pipeline")

        assert dashboard.state.data_trackers["test_pipeline"].step_count == initial_steps + 1
        assert dashboard.state.active_pipelines["test_pipeline"]["total_calls"] == initial_calls + 1

    def test_serialize_value(self):
        """Test value serialization for JSON transmission."""
        dashboard = InteractiveDashboard()

        # Test scalar values
        assert dashboard._serialize_value(42) == 42
        assert dashboard._serialize_value(3.14) == 3.14
        assert dashboard._serialize_value("test") == "test"
        assert dashboard._serialize_value(True)

        # Test small array
        small_array = jnp.array([1, 2, 3])
        result = dashboard._serialize_value(small_array)
        assert result == [1, 2, 3]

        # Test large array (should get summary)
        large_array = jnp.ones((100, 100))
        result = dashboard._serialize_value(large_array)
        assert isinstance(result, dict)
        assert result["type"] == "array"
        assert result["shape"] == (100, 100)

        # Test dictionary
        test_dict = {"a": 1, "b": 2.0}
        result = dashboard._serialize_value(test_dict)
        assert result == {"a": 1, "b": 2.0}

    @patch("wax.flax.visualization.interactive_dashboard.HAS_FLASK", False)
    def test_dashboard_without_flask(self):
        """Test dashboard behavior when Flask is not available."""
        dashboard = InteractiveDashboard()

        # Should handle gracefully
        assert dashboard.app is None
        assert dashboard.socketio is None

    def test_render_dashboard_html(self):
        """Test HTML dashboard rendering."""
        config = DashboardConfig(title="Test Dashboard", theme="dark")
        dashboard = InteractiveDashboard(config)

        html = dashboard._render_dashboard_html()

        assert "<!DOCTYPE html>" in html
        assert "Test Dashboard" in html
        assert "#2b2b2b" in html  # Dark theme color


class TestIntegration:
    """Integration tests combining multiple visualization components."""

    def test_end_to_end_visualization_workflow(self):
        """Test complete visualization workflow."""

        # Create a streaming function
        @streaming_transform_with_state
        def streaming_processor(x):
            ewma = EWMA(alpha=0.1)(x)
            buffer = Buffer(maxlen=10)(x)
            return {"ewma": ewma, "buffer": buffer}

        # Create data tracker
        tracker = DataFlowTracker()

        # Simulate streaming data
        rng = jax.random.PRNGKey(42)
        params, state = streaming_processor.init(rng, jnp.array(1.0))

        current_state = state
        for i in range(20):
            data_value = jnp.array(float(i))

            # Record input
            tracker.record_data("processor", "input", data_value)

            # Process data
            output, current_state = streaming_processor.apply(
                params, current_state, None, data_value
            )

            # Record output
            tracker.record_data("processor", "output", output)
            tracker.step()

        # Test graph rendering
        renderer = ComputationGraphRenderer(output_format="text")
        renderer.analyze_streaming_function(streaming_processor, jnp.array(1.0))
        graph_output = renderer.render()

        assert "WAX-ML Streaming Pipeline Graph" in graph_output

        # Test data visualization
        visualizer = DataFlowVisualizer(backend="text")
        visualizer.attach_tracker(tracker)
        plot_output = visualizer.create_streaming_plot()

        assert "WAX-ML Streaming Data Flow" in plot_output
        assert "processor" in plot_output
        assert "Total steps: 20" in plot_output

        # Test dashboard integration
        dashboard = InteractiveDashboard()
        dashboard.register_pipeline("test_pipeline", streaming_processor, jnp.array(1.0))

        # Simulate dashboard data recording
        for i in range(5):
            dashboard.record_pipeline_data(
                "test_pipeline",
                "processor",
                "input",
                jnp.array(float(i)),
                {"execution_time_ms": 2.0},
            )
            dashboard.step_pipeline("test_pipeline")

        assert len(dashboard.state.data_trackers["test_pipeline"].data_history) == 5

    def test_multiple_pipeline_monitoring(self):
        """Test monitoring multiple pipelines simultaneously."""
        dashboard = InteractiveDashboard()

        # Create different pipelines
        @streaming_transform_with_state
        def pipeline1(x):
            return EWMA(alpha=0.1)(x)

        @streaming_transform_with_state
        def pipeline2(x):
            return Buffer(maxlen=5)(x)

        # Register pipelines
        dashboard.register_pipeline("ewma_pipeline", pipeline1, jnp.array(1.0))
        dashboard.register_pipeline("buffer_pipeline", pipeline2, jnp.array([1.0, 2.0]))

        assert len(dashboard.state.active_pipelines) == 2
        assert len(dashboard.state.data_trackers) == 2

        # Record data for both pipelines
        dashboard.record_pipeline_data("ewma_pipeline", "ewma", "input", 1.0)
        dashboard.record_pipeline_data("buffer_pipeline", "buffer", "input", [1.0, 2.0])

        # Check data was recorded separately
        ewma_data = dashboard.state.data_trackers["ewma_pipeline"].get_recent_data()
        buffer_data = dashboard.state.data_trackers["buffer_pipeline"].get_recent_data()

        assert len(ewma_data) == 1
        assert len(buffer_data) == 1
        assert ewma_data[0].module_name == "ewma"
        assert buffer_data[0].module_name == "buffer"

    def test_performance_monitoring_and_alerts(self):
        """Test performance monitoring and alerting functionality."""
        config = DashboardConfig(
            enable_alerts=True, performance_threshold_ms=10.0, memory_threshold_mb=100.0
        )
        dashboard = InteractiveDashboard(config)

        @streaming_transform_with_state
        def test_pipeline(x):
            return EWMA(alpha=0.1)(x)

        dashboard.register_pipeline("test_pipeline", test_pipeline, jnp.array(1.0))

        # Record data that should trigger performance alert
        dashboard.record_pipeline_data(
            "test_pipeline",
            "ewma",
            "output",
            1.0,
            metadata={"execution_time_ms": 50.0},  # Above threshold
        )

        # Check that alert was generated
        assert len(dashboard.state.alerts) == 1
        alert = dashboard.state.alerts[0]
        assert alert["type"] == "performance"
        assert "50.0ms" in alert["message"]

        # Record data that should trigger memory alert
        dashboard.record_pipeline_data(
            "test_pipeline",
            "ewma",
            "output",
            1.0,
            metadata={"memory_usage_mb": 500.0},  # Above threshold
        )

        # Check that memory alert was generated
        assert len(dashboard.state.alerts) == 2
        memory_alert = dashboard.state.alerts[1]
        assert memory_alert["type"] == "memory"
        assert "500.0MB" in memory_alert["message"]
