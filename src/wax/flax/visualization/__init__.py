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
"""Visualization tools for WAX-ML streaming pipelines."""

from .computation_graph import (
    ComputationGraphRenderer,
    PipelineEdge,
    PipelineNode,
    export_graph_to_dot,
    render_pipeline_graph,
)
from .data_flow_visualizer import (
    DataFlowTracker,
    DataFlowVisualizer,
    create_flow_animation,
    visualize_streaming_data,
)
from .interactive_dashboard import (
    DashboardConfig,
    InteractiveDashboard,
    create_pipeline_dashboard,
    launch_monitoring_server,
)

__all__ = [
    "ComputationGraphRenderer",
    "PipelineNode",
    "PipelineEdge",
    "render_pipeline_graph",
    "export_graph_to_dot",
    "DataFlowVisualizer",
    "DataFlowTracker",
    "visualize_streaming_data",
    "create_flow_animation",
    "InteractiveDashboard",
    "DashboardConfig",
    "create_pipeline_dashboard",
    "launch_monitoring_server",
]

# Jupyter-specific visualizations (optional imports)
try:
    from .jupyter_viz import (
        AnimatedPipelineFlow,
        InteractiveParameterControls,
        InteractivePipelineGraph,
        JupyterVizConfig,
        StreamingDataVisualizer,
        create_pipeline_dashboard as create_jupyter_dashboard,
        display_pipeline_dashboard,
        quick_pipeline_viz,
        quick_streaming_plot,
    )
except ImportError:
    pass
else:
    __all__ += [
        "JupyterVizConfig",
        "InteractivePipelineGraph",
        "StreamingDataVisualizer",
        "InteractiveParameterControls",
        "AnimatedPipelineFlow",
        "create_jupyter_dashboard",
        "display_pipeline_dashboard",
        "quick_pipeline_viz",
        "quick_streaming_plot",
    ]

try:
    from .bokeh_viz import (
        BokehHeatmapVisualizer,
        BokehMultiPanelDashboard,
        BokehStreamingPlot,
        BokehVizConfig,
        create_bokeh_streaming_demo,
        display_bokeh_visualization,
    )
except ImportError:
    pass
else:
    __all__ += [
        "BokehVizConfig",
        "BokehStreamingPlot",
        "BokehHeatmapVisualizer",
        "BokehMultiPanelDashboard",
        "create_bokeh_streaming_demo",
        "display_bokeh_visualization",
    ]

# Nested graph visualizations (advanced graph visualization)
try:
    from .nested_graph_viz import (
        CytoscapeNestedRenderer,
        D3NestedRenderer,
        GraphHierarchy,
        GraphvizNestedRenderer,
        HierarchicalGraphAnalyzer,
        NestedGraphConfig,
        NestedGraphVisualizer,
        display_nested_graph_jupyter,
        visualize_nested_graph,
    )
except ImportError:
    pass
else:
    __all__ += [
        "NestedGraphConfig",
        "GraphHierarchy",
        "HierarchicalGraphAnalyzer",
        "GraphvizNestedRenderer",
        "CytoscapeNestedRenderer",
        "D3NestedRenderer",
        "NestedGraphVisualizer",
        "visualize_nested_graph",
        "display_nested_graph_jupyter",
    ]
