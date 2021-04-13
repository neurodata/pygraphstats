# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from graspologic.layouts.classes import NodePosition
import matplotlib.pyplot as plt


def _calculate_x_y_domain(
    positions: List[NodePosition],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """calculate the overall x/y domain, converting to a square
    so we can have a consistent scale
    """
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for node_position in positions:
        min_x = min(min_x, node_position.x - node_position.size)
        max_x = max(max_x, node_position.x + node_position.size)
        min_y = min(min_y, node_position.y - node_position.size)
        max_y = max(max_y, node_position.y + node_position.size)

    x_delta = max_x - min_x
    y_delta = max_y - min_y
    max_delta = max(x_delta, y_delta)

    if max_delta == x_delta:
        difference = (max_delta - y_delta) / 2
        min_y = min_y - difference
        max_y = max_y + difference
    elif max_delta == y_delta:
        difference = (max_delta - x_delta) / 2
        min_x = min_x - difference
        max_x = max_x + difference

    return (min_x, max_x), (min_y, max_y)


def _scale_value(
    domain: Tuple[float, float], data_range: Tuple[float, float], value: float
) -> float:
    return data_range[0] + (data_range[1] - data_range[0]) * (
        (value - domain[0]) / (domain[1] - domain[0])
    )


def _scale_node_sizes_for_rendering(
    sizes: List[float],
    spatial_domain: Tuple[float, float],
    spatial_range: Tuple[float, float],
    dpi: float,
) -> List[float]:
    """
    Scale the size again to match the rendered pixel range
    we would expect this to be handled by the underlying viz framework, but it isn't, size is specified
    as the bounding box in points of the rendered output, so we need to transform our size to match.

    There are 72 points per inch. Multiplying by 72 / dpi converts from pixels to points.
    """
    spatial_domain = (0, spatial_domain[1] - spatial_domain[0])
    return [
        _scale_value(spatial_domain, spatial_range, s * 2 * 72.0 / dpi) ** 2
        for s
        in sizes
    ]


def _draw_graph(
    graph: nx.Graph,
    positions: List[NodePosition],
    node_colors: Dict[Any, str],
    vertex_alpha: float,
    edge_line_width: float,
    edge_alpha: float,
    figure_width: float,
    figure_height: float,
    vertex_line_width: float = 0.01,
    vertex_shape: str = "o",
    arrows: bool = False,
    dpi: int = 100,
) -> None:
    if len(positions) != len(graph.nodes()):
        raise ValueError(
            f"The number of positions provided {len(positions)} is not the same as the "
            f"number of nodes in the graph {len(graph.nodes())}"
        )
    for position in positions:
        if position.node_id not in graph:
            raise ValueError(
                f"The node position provided for {position.node_id} references a node "
                f"not found in our graph"
            )

    plt.rcParams["figure.dpi"] = dpi  # TODO, test at different dpi

    plt.clf()
    figure = plt.gcf()
    ax = plt.gca()
    ax.set_axis_off()
    figure.set_size_inches(figure_width, figure_height)
    window_extent_width = ax.get_window_extent().width

    x_domain, y_domain = _calculate_x_y_domain(positions)

    position_map = {position.node_id: position for position in positions}
    node_positions = {
        position.node_id: (position.x, position.y) for position in positions
    }

    vertices = []
    vertex_sizes = []
    node_color_list = []
    edge_color_list = []

    for node in graph.nodes():
        vertices.append(node)
        vertex_sizes.append(position_map[node].size)
        node_color_list.append(node_colors[node])

    vertex_sizes = _scale_node_sizes_for_rendering(
        vertex_sizes, x_domain, (0, window_extent_width), dpi
    )

    for source, target in graph.edges():
        edge_color_list.append(node_colors[source])

    ax.set_xbound(x_domain)
    ax.set_xlim(x_domain)
    ax.set_ybound(y_domain)
    ax.set_ylim(y_domain)

    nx.draw_networkx_edges(
        graph,
        pos=node_positions,
        alpha=edge_alpha,
        width=edge_line_width,
        edge_color=edge_color_list,
        arrows=arrows,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        graph,
        pos=node_positions,
        nodelist=vertices,
        node_color=node_color_list,
        alpha=vertex_alpha,
        linewidths=vertex_line_width,
        node_size=vertex_sizes,
        node_shape=vertex_shape,
        ax=ax,
    )


def show_graph(
    graph: nx.Graph,
    positions: List[NodePosition],
    node_colors: Dict[Any, str],
    vertex_line_width: float = 0.01,
    vertex_alpha: float = 0.55,
    edge_line_width: float = 0.5,
    edge_alpha: float = 0.02,
    figure_width: float = 15.0,
    figure_height: float = 15.0,
    light_background: bool = True,
    vertex_shape: str = "o",
    arrows: bool = False,
    dpi: int = 500,
) -> None:
    """
    Renders and displays a graph.

    Attempts to display it via the platform-specific display library such as TkInter

    Edges will be displayed with the same color as the source node.

    Parameters
    ----------
    graph : nx.Graph
        The graph to be displayed. If the networkx Graph contains only nodes, no
        edges will be displayed.
    positions : List[:class:`graspologic.layouts.NodePosition`]
        The positionsfor every node in the graph.
    node_colors : Dict[Any, str]
        A mapping of node id to colors. Must contain an entry for every node in the
        graph.
    vertex_line_width : float
        Line width of vertex outline. Default is``0.01``.
    vertex_alpha : float
        Alpha (transparency) of vertices in visualization. Default is``0.55``.
    edge_line_width : float
        Line width of edge. Default is``0.5``.
    edge_alpha : float
        Alpha (transparency) of edges in visualization. Default is``0.02``.
    figure_width : float
        Width of figure. Default is ``15.0``.
    figure_height : float
        eight of figure. Default is``15.0``.
    light_background : bool
        Light background or dark background. Default is``True``.
    vertex_shape : str
        Matplotlib Marker for the vertex shape.  See
        `https://matplotlib.org/api/markers_api.html <https://matplotlib.org/api/markers_api.html>`_
        for a list of allowed values . Default is ``o`` (i.e: a circle)
    arrows : bool
        For directed graphs, if ``True``, draw arrow heads. Default is ``False``
    dpi : float
        Dots per inch of the figure.  Default is ``500``.
    """
    ax = plt.gca()
    if light_background:
        facecolor = ax.get_facecolor()
    else:
        facecolor = "#030303"

    _draw_graph(
        graph=graph,
        positions=positions,
        node_colors=node_colors,
        vertex_line_width=vertex_line_width,
        vertex_alpha=vertex_alpha,
        edge_line_width=edge_line_width,
        edge_alpha=edge_alpha,
        figure_width=figure_width,
        figure_height=figure_height,
        vertex_shape=vertex_shape,
        arrows=arrows,
        dpi=dpi,
    )
    plt.gcf().set_facecolor(facecolor)
    plt.show()
    plt.close("all")


def save_graph(
    output_path: str,
    graph: nx.Graph,
    positions: List[NodePosition],
    node_colors: Dict[Any, str],
    vertex_line_width: float = 0.01,
    vertex_alpha: float = 0.55,
    edge_line_width: float = 0.5,
    edge_alpha: float = 0.02,
    figure_width: float = 15.0,
    figure_height: float = 15.0,
    light_background: bool = True,
    vertex_shape: str = "o",
    arrows: bool = False,
    dpi: int = 100,
) -> None:
    """
    Renders a graph to file.

    Edges will be displayed with the same color as the source node.

    Parameters
    ----------
    output_path : str
        The output path to write the rendered graph to. Suggested file extension is
        ``.png``.
    graph : nx.Graph
        The graph to be displayed. If the networkx Graph contains only nodes, no
        edges will be displayed.
    positions : List[:class:`graspologic.layouts.NodePosition`]
        The positionsfor every node in the graph.
    node_colors : Dict[Any, str]
        A mapping of node id to colors. Must contain an entry for every node in the
        graph.
    vertex_line_width : float
        Line width of vertex outline. Default is``0.01``.
    vertex_alpha : float
        Alpha (transparency) of vertices in visualization. Default is``0.55``.
    edge_line_width : float
        Line width of edge. Default is``0.5``.
    edge_alpha : float
        Alpha (transparency) of edges in visualization. Default is``0.02``.
    figure_width : float
        Width of figure. Default is ``15.0``.
    figure_height : float
        eight of figure. Default is``15.0``.
    light_background : bool
        Light background or dark background. Default is``True``.
    vertex_shape : str
        Matplotlib Marker for the vertex shape.  See
        `https://matplotlib.org/api/markers_api.html <https://matplotlib.org/api/markers_api.html>`_
        for a list of allowed values . Default is ``o`` (i.e: a circle)
    arrows : bool
        For directed graphs, if ``True``, draw arrow heads. Default is ``False``
    dpi : float
        Dots per inch of the figure.  Default is ``100``.

    Returns
    -------

    """
    _draw_graph(
        graph=graph,
        positions=positions,
        node_colors=node_colors,
        vertex_line_width=vertex_line_width,
        vertex_alpha=vertex_alpha,
        edge_line_width=edge_line_width,
        edge_alpha=edge_alpha,
        figure_width=figure_width,
        figure_height=figure_height,
        vertex_shape=vertex_shape,
        arrows=arrows,
        dpi=dpi,
    )
    ax = plt.gca()
    if light_background:
        facecolor = ax.get_facecolor()
    else:
        facecolor = "#030303"
    plt.savefig(output_path, facecolor=facecolor)
    plt.close("all")
