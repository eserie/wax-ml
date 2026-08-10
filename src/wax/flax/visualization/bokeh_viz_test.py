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
"""Tests that execute the bokeh rendering paths.

Until the `visualization` extra existed, bokeh was installed nowhere, so every test
touching this module walked the "bokeh missing" branch and the real code was never
run -- which is how it drifted out of step with the bokeh 3.x API unnoticed. These
tests only run when bokeh is importable, and they call the rendering code for real.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

from wax.flax.visualization import bokeh_viz
from wax.flax.visualization.bokeh_viz import (
    HAS_BOKEH,
    BokehHeatmapVisualizer,
    BokehMultiPanelDashboard,
    BokehStreamingPlot,
    BokehVizConfig,
)
from wax.flax.visualization.data_flow_visualizer import DataFlowTracker

requires_bokeh = pytest.mark.skipif(not HAS_BOKEH, reason="bokeh is not installed")


@requires_bokeh
class TestBokehRendering:
    """Exercise the bokeh code paths end to end."""

    def test_streaming_line_plot_is_configured_as_requested(self):
        config = BokehVizConfig(plot_width=321, plot_height=123, webgl=False)
        viz = BokehStreamingPlot(config)

        plot = viz.create_streaming_line_plot(title="T", x_label="X", y_label="Y")

        assert plot.width == 321
        assert plot.height == 123
        assert plot.output_backend == "canvas"
        assert plot.title.text == "T"
        assert plot.title.text_font_size == "14pt"
        assert plot.xaxis[0].axis_label == "X"
        assert plot.yaxis[0].axis_label == "Y"
        # The `tools` string is one of the options mypy cannot see on `figure`; the
        # only way to know it still reaches bokeh is to look at the built toolbar.
        assert {type(tool).__name__ for tool in plot.toolbar.tools} >= {
            "PanTool",
            "WheelZoomTool",
            "ResetTool",
        }

    def test_creating_a_plot_does_not_install_a_notebook_shell(self):
        """Building a figure must not mutate process-global IPython state.

        `create_streaming_line_plot` used to call `output_notebook()`, which
        instantiates a bare `InteractiveShell`. matplotlib's
        `install_repl_displayhook` then calls `enable_gui` on it -- unimplemented on
        the base shell -- so any later `DataFrame.plot()` in the same process died
        with `NotImplementedError`.
        """
        import IPython

        assert IPython.get_ipython() is None, "a previous test already leaked a shell"

        BokehStreamingPlot().create_streaming_line_plot()

        assert IPython.get_ipython() is None

    def test_add_line_stream_creates_line_and_marker_renderers(self):
        viz = BokehStreamingPlot()
        plot = viz.create_streaming_line_plot()

        source = viz.add_line_stream(plot, "signal", color="#123456")

        assert viz.data_sources["signal"] is source
        glyphs = {type(renderer.glyph).__name__ for renderer in plot.renderers}
        assert "Line" in glyphs
        # bokeh 3.4 deprecated `circle(size=...)`; markers now come from `scatter`.
        assert "Scatter" in glyphs

    def test_update_stream_appends_and_truncates(self):
        viz = BokehStreamingPlot(BokehVizConfig(max_points=3))
        plot = viz.create_streaming_line_plot()
        viz.add_line_stream(plot, "signal")

        for step in range(5):
            viz.update_stream("signal", float(step), float(step) * 2.0)

        data = viz.data_sources["signal"].data
        assert list(data["x"]) == [2.0, 3.0, 4.0]
        assert list(data["y"]) == [4.0, 6.0, 8.0]

    def test_update_stream_warns_on_unknown_stream(self):
        viz = BokehStreamingPlot()

        with pytest.warns(UserWarning, match="not found"):
            viz.update_stream("absent", 1.0, 2.0)

    def test_state_heatmap_covers_every_cell(self):
        data = np.arange(6.0).reshape(2, 3)

        plot = BokehHeatmapVisualizer().create_state_heatmap(data, title="H")

        assert plot.title.text == "H"
        rects = [r for r in plot.renderers if type(r.glyph).__name__ == "Rect"]
        assert len(rects) == 1
        assert len(rects[0].data_source.data["values"]) == data.size

    def test_multi_panel_dashboard_builds_all_four_panels(self):
        tracker = DataFlowTracker()
        for step in range(5):
            tracker.record_data("ewma", "output", float(step))
            tracker.step()

        dashboard = BokehMultiPanelDashboard()
        layout = dashboard.create_dashboard(tracker, layout_type="grid")

        assert set(dashboard.panels) == {
            "time_series",
            "distribution",
            "performance",
            "correlation",
        }
        assert dashboard.layout is layout
        assert type(layout).__name__ == "GridPlot"

    @pytest.mark.parametrize(("layout_type", "expected"), [("column", "Column"), ("row", "Row")])
    def test_multi_panel_dashboard_honours_layout_type(self, layout_type, expected):
        layout = BokehMultiPanelDashboard().create_dashboard(
            DataFlowTracker(), layout_type=layout_type
        )

        assert type(layout).__name__ == expected


class TestBokehGuard:
    """The guard must stay usable on a plain `wax-ml` install."""

    def test_constructing_without_bokeh_warns_and_names_the_extra(self):
        with (
            patch.object(bokeh_viz, "HAS_BOKEH", False),
            pytest.warns(UserWarning, match=r"wax-ml\[visualization\]"),
        ):
            BokehStreamingPlot()

    @pytest.mark.parametrize(
        "call",
        [
            lambda: BokehStreamingPlot().create_streaming_line_plot(),
            lambda: BokehHeatmapVisualizer().create_state_heatmap(np.zeros((2, 2))),
            lambda: BokehMultiPanelDashboard().create_dashboard(DataFlowTracker()),
            lambda: bokeh_viz.create_bokeh_streaming_demo(DataFlowTracker()),
            lambda: bokeh_viz.display_bokeh_visualization(None),
        ],
    )
    def test_using_a_feature_without_bokeh_raises_naming_the_extra(self, call):
        # The constructors also warn; that is covered above, so ignore it here.
        with (
            warnings.catch_warnings(),
            patch.object(bokeh_viz, "HAS_BOKEH", False),
            pytest.raises(ImportError, match=r"Bokeh is required for .*wax-ml\[visualization\]"),
        ):
            warnings.simplefilter("ignore")
            call()
