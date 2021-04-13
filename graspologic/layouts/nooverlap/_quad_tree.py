# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import List

from ._node import _Node
from ._quad_node import _QuadNode


class _QuadTree:
    # used to hold objects that have x, y, and mass property
    # nodes = []

    def __init__(
        self,
        nodes: List[_Node],
        max_nodes_per_quad: int
    ):
        self.nodes = nodes
        self.root = _QuadNode(nodes, 0, max_nodes_per_quad, None)

    def get_quad_density_list(self):
        density_list = self.root.get_density_list()
        return sorted(density_list, reverse=True)

    def layout_graph(self):
        return self.layout_dense_first()

    def tree_stats(self):
        results = self.root.quad_stats()
        return list(results) + [
            results[3] / len(self.nodes),
            results[4] / len(self.nodes),
            self.root.sq_ratio,
        ]

    def collect_nodes(self):
        ret_val = []
        self.root.collect_nodes(ret_val)
        return ret_val

    def get_tree_node_bounds(self):
        ret_val = []
        self.root.boxes_by_level(ret_val)
        return ret_val

    def count_overlaps(self):
        return self.root.num_overlapping()

    def count_overlaps_across_quads(self):
        return self.root.num_overlapping_across_quads(self.root.nodes)

    def layout_dense_first(self, first_color=None):
        den_list = list(self.get_quad_density_list())
        first = True
        # count = 0
        for cell_density, density_ratio, cell_count, qn in den_list:
            # print ('cell density', cell_density, 'sq_density', density_ratio, 'cell_count', cell_count)
            qn.layout_quad()
            if first:
                if first_color is not None:
                    for n in qn.parent.nodes:
                        n.color = first_color  #'#FF0004'
            first = False
        return self.nodes
