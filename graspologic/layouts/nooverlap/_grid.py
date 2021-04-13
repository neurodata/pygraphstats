# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from ._node import _Node


class _GridBuckets:
    """
    One thing to note, that right now this grid must have cells the same size or bigger
    than the radius of the largest node in the graph.
    """

    def __init__(self, cell_size: int):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[_Node]] = defaultdict(list)
        self.min_x = math.inf
        self.min_y = math.inf
        self.max_x = -math.inf
        self.max_y = -math.inf

    def get_cell(self, x: float, y: float) -> Tuple[int, int]:
        x_val = int(x // self.cell_size)
        y_val = int(y // self.cell_size)
        return x_val * self.cell_size, y_val * self.cell_size

    def get_grid_cells(
        self,
        x: float,
        y: float,
        node_size: float
    ) -> Set[Tuple[int, int]]:
        return self._get_grid_cells(x, y, node_size, False)

    def _get_grid_cells(
        self,
        x: float,
        y: float,
        node_size: float,
        update_max: bool = True
    ) -> Set[Tuple[int, int]]:
        """
        Each node will be at least one cell but up to four cells.
        It depends on the grid cell size and the size and location of the node
        :param x: x location
        :param y: y location
        :param node_size: nodesize
        :return:
        """
        min_x = x - node_size
        max_x = x + node_size
        min_y = y - node_size
        max_y = y + node_size
        if update_max:
            self.min_x = min(self.min_x, min_x)
            self.min_y = min(self.min_y, min_y)
            self.max_x = max(self.max_x, max_x)
            self.max_y = max(self.max_y, max_y)
        nw = self.get_cell(min_x, max_y)
        ne = self.get_cell(max_x, max_y)
        se = self.get_cell(max_x, min_y)
        sw = self.get_cell(min_x, min_y)
        return {nw, ne, se, sw}

    def add_node(self, node: _Node) -> _Node:
        """
        Node to add must have an x, y, and size property
        :param node: The node to add to the grid
        :return: The node added
        """
        cells = self._get_grid_cells(node.x, node.y, node.size)
        for cell in cells:
            self.grid[cell].append(node)
        return node

    def add_node_once(self, node: _Node) -> _Node:
        """
        Node to add must have an x, y, and size property
        :param node: The node to add to the grid
        :return: The node added
        """
        cell = self.get_cell(node.x, node.y)
        self.grid[cell].append(node)
        return node

    def remove_node(self, node):
        """
        Will throw an exception if the node is not in the grid.
        :param node:  node to remove
        :return: the node removed
        """
        cells = self._get_grid_cells(node.x, node.y, node.size)
        for cell in cells:
            self.grid[cell].remove(node)
        return node

    def get_nodes_for_cell(self, cell):
        return self.grid[cell]

    def get_potential_overlapping_nodes_by_node(self, node):
        return self.get_potential_overlapping_nodes(node.x, node.y, node.size)

    def get_potential_overlapping_nodes(self, x, y, size):
        cells = self._get_grid_cells(x, y, size, update_max=False)
        nodes = set()
        for c in cells:
            nodes.update(self.grid[c])
        return nodes

    def _get_x_cells(self):
        xrange = self.max_x - self.min_x
        return xrange // self.cell_size

    def _get_y_cells(self):
        yrange = self.max_y - self.min_y
        return yrange // self.cell_size

    def num_cells(self):
        x_cells = self._get_x_cells()
        y_cells = self._get_y_cells()
        return x_cells * y_cells

    def add_nodes(self, node_list, only_once=False):
        for n in node_list:
            if only_once:
                self.add_node_once(n)
            else:
                self.add_node(n)
        return self

    def max_in_cell(self):
        max_in_cell = -1
        for c in self.grid:
            max_in_cell = max(max_in_cell, len(self.grid[c]))
        return max_in_cell

    def get_grid_cell_stats(self, with_zeros=True):
        num_nodes_in_cell_to_count_of_cells = defaultdict(int)
        for c in self.grid:
            num_nodes_in_cell_to_count_of_cells[len(self.grid[c])] += 1
        if with_zeros:
            number_of_cells = self.num_cells()
            number_of_zeros = int(number_of_cells - len(self.grid))
            num_nodes_in_cell_to_count_of_cells[0] = number_of_zeros
        return [
            (cell_count, node_count)
            for cell_count, node_count in sorted(
                num_nodes_in_cell_to_count_of_cells.items()
            )
        ]

    def get_grid_bounds(self):
        x_min, y_min = math.inf, math.inf
        x_max, y_max = -math.inf, -math.inf
        for x, y in self.grid:
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        return x_min, y_min, x_max, y_max

    def get_all_grid_cells(self):
        rows = []
        x_min, y_min, x_max, y_max = self.get_grid_bounds()
        cell_size = self.cell_size
        # print (type(y_min), type(y_max), type(cell_size))
        y = y_min
        while y <= y_max:
            row = []
            x = x_min
            while x <= x_max:
                row.append(len(self.grid[(x, y)]))
                x += cell_size
            rows.insert(0, row)
            y += cell_size
        # for c in sorted(self.grid):
        # 	print (c)
        return rows

    def print_stats(self):
        print(
            f"cell size: {self.cell_size}, area: {self.cell_size*self.cell_size}, "
            f"rows: {self._get_y_cells()}, cols: {self._get_x_cells()}"
        )
        return None

    def _get_area(self, node_list):
        area = 0.0
        for n in node_list:
            area += n.size * n.size * 4
        return area

    def get_all_grid_cells_by_area(self):
        rows = []
        x_min, y_min, x_max, y_max = self.get_grid_bounds()
        cell_size = self.cell_size
        # print (type(y_min), type(y_max), type(cell_size))
        y = y_min
        while y <= y_max:
            row = []
            x = x_min
            while x <= x_max:
                row.append(self._get_area(self.grid[(x, y)]))
                x += cell_size
            rows.insert(0, row)
            y += cell_size
        # for c in sorted(self.grid):
        # 	print (c)
        return rows
