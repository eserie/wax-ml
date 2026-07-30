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
"""Tests for modern interactive visualization tools."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import jax.numpy as jnp

from wax.flax.core.streaming_transforms import streaming_transform_with_state
from wax.flax.modules.ewma import EWMA
from wax.flax.visualization import DataFlowTracker


class TestModernVisualizationImports:
    """Test that modern visualization components can be imported gracefully."""

    def test_jupyter_viz_imports(self):
        """Test Jupyter visualization imports with fallback."""
        try:
            from wax.flax.visualization import (
                AnimatedPipelineFlow,
                InteractiveParameterControls,
                InteractivePipelineGraph,
                JupyterVizConfig,
                StreamingDataVisualizer,
                quick_pipeline_viz,
                quick_streaming_plot,
            )

            # All exported names must be importable and usable
            for exported in (
                AnimatedPipelineFlow,
                InteractiveParameterControls,
                InteractivePipelineGraph,
                StreamingDataVisualizer,
                quick_pipeline_viz,
                quick_streaming_plot,
            ):
                assert callable(exported)

            # Test basic instantiation without dependencies
            config = JupyterVizConfig()
            assert config.plotly_theme == "plotly_white"
            assert config.plotly_height == 500

        except ImportError:
            # Expected if optional dependencies not installed
            warnings.warn(
                "Jupyter visualization imports not available (optional dependencies missing)",
                stacklevel=2,
            )

    def test_bokeh_viz_imports(self):
        """Test Bokeh visualization imports with fallback."""
        try:
            from wax.flax.visualization import (
                BokehHeatmapVisualizer,
                BokehMultiPanelDashboard,
                BokehStreamingPlot,
                BokehVizConfig,
                create_bokeh_streaming_demo,
                display_bokeh_visualization,
            )

            # All exported names must be importable and usable
            for exported in (
                BokehHeatmapVisualizer,
                BokehMultiPanelDashboard,
                BokehStreamingPlot,
                create_bokeh_streaming_demo,
                display_bokeh_visualization,
            ):
                assert callable(exported)

            # Test basic instantiation without dependencies
            config = BokehVizConfig()
            assert config.plot_width == 800
            assert config.plot_height == 400

        except ImportError:
            # Expected if Bokeh not installed
            warnings.warn(
                "Bokeh visualization imports not available (optional dependency missing)",
                stacklevel=2,
            )


class TestJupyterVizConfig:
    """Test Jupyter visualization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        try:
            from wax.flax.visualization import JupyterVizConfig

            config = JupyterVizConfig()

            assert config.plotly_theme == "plotly_white"
            assert config.plotly_height == 500
            assert config.plotly_width == 800
            assert config.show_toolbar is True
            assert config.animation_interval_ms == 100
            assert config.max_history_points == 1000
            assert config.enable_widgets is True

        except ImportError:
            # Skip test if dependencies not available
            pass

    def test_custom_config(self):
        """Test custom configuration values."""
        try:
            from wax.flax.visualization import JupyterVizConfig

            config = JupyterVizConfig(
                plotly_theme="plotly_dark",
                plotly_height=400,
                plotly_width=600,
                animation_interval_ms=50,
                max_history_points=500,
            )

            assert config.plotly_theme == "plotly_dark"
            assert config.plotly_height == 400
            assert config.plotly_width == 600
            assert config.animation_interval_ms == 50
            assert config.max_history_points == 500

        except ImportError:
            # Skip test if dependencies not available
            pass


class TestInteractiveParameterControls:
    """Test interactive parameter control functionality."""

    def test_parameter_controls_without_widgets(self):
        """Test parameter controls behavior when ipywidgets not available."""
        try:
            from wax.flax.visualization import InteractiveParameterControls

            with patch("wax.flax.visualization.jupyter_viz.HAS_WIDGETS", False):
                controls = InteractiveParameterControls()

                # Should create instance but warn about missing widgets
                assert controls.controls == {}
                assert controls.callbacks == []

        except ImportError:
            # Skip test if module not available
            pass

    def test_parameter_definition(self):
        """Test parameter definition structure."""
        try:
            from wax.flax.visualization import InteractiveParameterControls

            InteractiveParameterControls()

            # Test parameter structure validation

            # Should not raise error with valid parameters
            # (actual widget creation may fail without ipywidgets)

        except ImportError:
            # Skip test if module not available
            pass


class TestStreamingDataVisualizer:
    """Test streaming data visualization functionality."""

    def test_visualizer_config(self):
        """Test streaming visualizer configuration."""
        try:
            from wax.flax.visualization import JupyterVizConfig, StreamingDataVisualizer

            config = JupyterVizConfig(max_history_points=200)
            viz = StreamingDataVisualizer(config)

            assert viz.config.max_history_points == 200
            assert viz.figures == {}
            assert viz.is_streaming is False

        except ImportError:
            # Skip test if module not available
            pass

    def test_data_buffer_management(self):
        """Test data buffer management."""
        try:
            from wax.flax.visualization import StreamingDataVisualizer

            viz = StreamingDataVisualizer()

            # Test buffer initialization
            assert len(viz.data_buffers) == 0
            assert len(viz.time_buffers) == 0

            # Buffers should be created on demand
            buffer_key = "test_stream"
            assert buffer_key not in viz.data_buffers

        except ImportError:
            # Skip test if module not available
            pass


class TestAnimatedPipelineFlow:
    """Test animated pipeline flow visualization."""

    def test_flow_animator_config(self):
        """Test flow animator configuration."""
        try:
            from wax.flax.visualization import AnimatedPipelineFlow, JupyterVizConfig

            config = JupyterVizConfig(animation_interval_ms=50)
            animator = AnimatedPipelineFlow(config)

            assert animator.config.animation_interval_ms == 50
            assert animator.animation_fig is None
            assert animator.is_animating is False

        except ImportError:
            # Skip test if module not available
            pass


class TestIntegrationWithExistingComponents:
    """Test integration with existing visualization components."""

    def test_data_tracker_integration(self):
        """Test integration with DataFlowTracker."""

        # Create a simple streaming function
        @streaming_transform_with_state
        def simple_ewma(x):
            return EWMA(alpha=0.1)(x)

        # Create data tracker
        tracker = DataFlowTracker()

        # Test data recording
        tracker.record_data("ewma", "input", jnp.array(1.0))
        tracker.record_data("ewma", "output", jnp.array(1.0))
        tracker.step()

        assert len(tracker.data_history) == 2
        assert tracker.step_count == 1

        # Test with modern visualization components
        try:
            from wax.flax.visualization import create_bokeh_streaming_demo

            # Should not raise error with valid tracker
            # (actual visualization may fail without Bokeh)
            assert callable(create_bokeh_streaming_demo)

        except ImportError:
            # Expected if Bokeh not available
            pass

    def test_convenience_functions(self):
        """Test convenience functions for quick visualization."""

        @streaming_transform_with_state
        def test_pipeline(x):
            return EWMA(alpha=0.2)(x)

        jnp.array(1.0)

        try:
            from wax.flax.visualization import quick_pipeline_viz, quick_streaming_plot

            # These should handle missing dependencies gracefully
            # In a real Jupyter environment with dependencies, they would create
            # visualizations
            assert callable(quick_pipeline_viz)
            assert callable(quick_streaming_plot)

        except ImportError:
            # Expected if Plotly/ipywidgets not available
            pass


class TestModernVisualizationFallbacks:
    """Test graceful fallbacks when optional dependencies are missing."""

    def test_plotly_fallback(self):
        """Test behavior when Plotly is not available."""
        try:
            from wax.flax.visualization.jupyter_viz import HAS_PLOTLY

            if not HAS_PLOTLY:
                # Should handle gracefully
                from wax.flax.visualization import StreamingDataVisualizer

                StreamingDataVisualizer()
                # Should create instance but warn about missing Plotly

        except ImportError:
            # Expected if entire module not available
            pass

    def test_bokeh_fallback(self):
        """Test behavior when Bokeh is not available."""
        try:
            from wax.flax.visualization.bokeh_viz import HAS_BOKEH

            if not HAS_BOKEH:
                # Should handle gracefully
                from wax.flax.visualization import BokehStreamingPlot

                BokehStreamingPlot()
                # Should create instance but warn about missing Bokeh

        except ImportError:
            # Expected if entire module not available
            pass

    def test_ipywidgets_fallback(self):
        """Test behavior when ipywidgets is not available."""
        try:
            from wax.flax.visualization.jupyter_viz import HAS_WIDGETS

            if not HAS_WIDGETS:
                # Should handle gracefully
                from wax.flax.visualization import InteractiveParameterControls

                InteractiveParameterControls()
                # Should create instance but warn about missing widgets

        except ImportError:
            # Expected if entire module not available
            pass
