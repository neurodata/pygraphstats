# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import networkx as nx

from . import auto, NodePosition, render
from .colors import categorical_colors


def _graph_from_file(
    path: str,
    skip_header: bool = False,
) -> nx.Graph:
    graph = nx.Graph()
    with open(path, "r") as edge_io:
        if skip_header is True:
            next(edge_io)
        for line in edge_io:
            source, target, weight_str = line.strip().split(",")
            weight: float = float(weight_str)
            if graph.has_edge(source, target):
                weight += graph[source][target]["weight"]
            graph.add_edge(source, target, weight=weight)
    return graph


def _ensure_output_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _location(path: str, positions: List[NodePosition], colors: Dict[Any, str]) -> None:
    with open(path, "w") as node_positions_out:
        print("id,x,y,size,community,color", file=node_positions_out)
        for position in positions:
            print(
                f"{position.node_id},{position.x},{position.y},{position.size},"
                f"{position.community},{colors[position.node_id]}",
                file=node_positions_out,
            )


def _output(
    arguments: argparse.Namespace, graph: nx.Graph, positions: List[NodePosition]
) -> None:
    partitions = {position.node_id: position.community for position in positions}
    color_map = categorical_colors(partitions)
    if arguments.image_file is not None:
        render.save_graph(arguments.image_file, graph, positions, node_colors=color_map)
    if arguments.location_file is not None:
        _location(arguments.location_file, positions, color_map)


def _tsne(arguments: argparse.Namespace) -> None:
    valid_args(arguments)
    graph = _graph_from_file(arguments.edge_list, arguments.skip_header)
    adjust_overlaps = not arguments.allow_overlaps
    graph, positions = auto.layout_tsne(
        graph,
        perplexity=30,
        n_iter=1000,
        max_edges=arguments.max_edges,
        adjust_overlaps=adjust_overlaps,
    )
    _output(arguments, graph, positions)


def _umap(arguments: argparse.Namespace) -> None:
    valid_args(arguments)
    graph = _graph_from_file(arguments.edge_list, arguments.skip_header)
    adjust_overlaps = not arguments.allow_overlaps
    graph, positions = auto.layout_umap(
        graph, max_edges=arguments.max_edges, adjust_overlaps=adjust_overlaps
    )
    _output(arguments, graph, positions)


def _render(arguments: argparse.Namespace) -> None:
    positions = []
    node_colors = {}
    with open(arguments.location_file, "r") as location_io:
        next(location_io)
        for line in location_io:
            node_id, x, y, size, community, color = line.strip().split(",")
            positions.append(
                NodePosition(node_id, float(x), float(y), float(size), int(community))
            )
            node_colors[node_id] = color
    if arguments.edge_list is not None:
        graph = _graph_from_file(arguments.edge_list, arguments.skip_header)
    else:
        graph = nx.Graph()
        for position in positions:
            graph.add_node(position.node_id)

    render.save_graph(
        arguments.image_file,
        graph,
        positions,
        node_colors=node_colors,
        vertex_line_width=arguments.vertex_line_width,
        vertex_alpha=arguments.vertex_alpha,
        edge_line_width=arguments.edge_line_width,
        edge_alpha=arguments.edge_alpha,
        figure_height=arguments.figure_height,
        figure_width=arguments.figure_width,
        vertex_shape=arguments.vertex_shape,
        arrows=arguments.arrows,
        dpi=arguments.dpi,
        light_background=arguments.light_background,
    )


def _common_edge_list_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--edge_list",
        help="edge list in csv file. must be source,target,weight.",
        required=True,
    )
    parser.add_argument(
        "--skip_header",
        help="skip first line in csv file, corresponding to header.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--image_file",
        help="output path and filename for generated image file. "
        "required if --location_file is omitted.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--location_file",
        help="output path and filename for location file. "
        "required if --image_file is omitted.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--max_edges",
        help="maximum edges to keep during embedding. edges with low weights will be "
        "pruned to keep at most this many edges",
        type=int,
        required=False,
        default=10000000,
    )
    parser.add_argument(
        "--dpi",
        help="used with --image_file to render an image at this dpi",
        type=int,
        required=False,
        default=500,
    )
    parser.add_argument(
        "--allow_overlaps",
        help="skip the no overlap algorithm and let nodes stack as per the results of "
        "the down projection algorithm",
        action="store_true",
    )
    return parser


def _parser() -> argparse.ArgumentParser:
    root_parser = argparse.ArgumentParser(
        prog="python -m graspologic.layouts",
        description="Runnable module that automatically generates a layout of a graph "
        "by a provided edge list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    root_parser.add_argument(
        "--verbose",
        type=bool,
        required=False,
        default=False,
    )

    subparsers = root_parser.add_subparsers(
        required=True,
        dest="COMMAND",
        help="auto layout via umap, tsne, or a pure render only mode",
    )

    n2vumap_parser = subparsers.add_parser(
        "n2vumap",
        help="Auto layout using UMAP for dimensionality reduction",
    )
    _common_edge_list_args(n2vumap_parser)
    n2vumap_parser.set_defaults(func=_umap)
    n2vtsne_parser = subparsers.add_parser(
        "n2vtsne",
        help="Auto layout using tSNE for dimensionality reduction",
    )
    _common_edge_list_args(n2vtsne_parser)
    n2vtsne_parser.set_defaults(func=_tsne)
    render_parser = subparsers.add_parser(
        "render",
        help="Renders a graph via an input file",
    )
    render_parser.set_defaults(func=_render)
    render_parser.add_argument(
        "--edge_list",
        help="edge list in csv file. must be source,target,weight.",
        required=False,
        default=None,
    )
    render_parser.add_argument(
        "--skip_header",
        help="skip first line in csv file, corresponding to header.",
        action="store_true",
        required=False,
        default=False,
    )
    render_parser.add_argument(
        "--location_file",
        help="location file used for node positioning, partitioning, and coloring",
        required=True,
        default=None,
    )
    render_parser.add_argument(
        "--image_file",
        help="output path and filename for generated image file. ",
        required=True,
        default=None,
    )
    render_parser.add_argument(
        "--dark_background",
        dest="light_background",
        action="store_false",
        default=True,
    )
    render_parser.add_argument(
        "--vertex_alpha",
        help="Alpha (transparency) of vertices in visualization (default 0.9)",
        type=float,
        required=False,
        default=0.9,
    )
    render_parser.add_argument(
        "--vertex_line_width",
        help="Line width of vertex outline (default 0.01)",
        type=float,
        required=False,
        default=0.01,
    )
    render_parser.add_argument(
        "--edge_line_width",
        help="Line width of edge (default 0.5)",
        type=float,
        required=False,
        default=0.5,
    )
    render_parser.add_argument(
        "--edge_alpha",
        help="Alpha (transparency) of edges in visualization (default 0.2)",
        type=float,
        required=False,
        default=0.2,
    )
    render_parser.add_argument(
        "--figure_width",
        help="Width of figure (default 15.0)",
        type=float,
        required=False,
        default=15.0,
    )
    render_parser.add_argument(
        "--figure_height",
        help="Height of figure (default 15.0)",
        type=float,
        required=False,
        default=15.0,
    )
    render_parser.add_argument(
        "--vertex_shape",
        help="Matplotlib Marker for the vertex shape.",
        required=False,
        default="o",
    )
    render_parser.add_argument(
        "--arrows",
        dest="arrows",
        action="store_true",
        default=False,
    )
    render_parser.add_argument(
        "--dpi",
        help="Set dpi for image",
        type=int,
        required=False,
        default=500,
    )
    return root_parser


def valid_args(args: argparse.Namespace) -> None:
    if args.image_file is None and args.location_file is None:
        print(
            "error: --image_file or --location_file must be provided", file=sys.stderr
        )
        exit(-1)
    if args.location_file is not None:
        _ensure_output_dir(args.location_file)
    if args.image_file is not None:
        _ensure_output_dir(args.image_file)


def main() -> None:
    parser = _parser()

    args = parser.parse_args()

    if args.verbose is True:
        logging.basicConfig(
            format="%(asctime)s:%(levelname)s:%(name)s, %(message)s", level=logging.INFO
        )
        logger = logging.getLogger("graspologic.layouts")

    args.func(args)


if __name__ == "__main__":
    main()
