# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
from typing import List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import normalize

from ._node import _Node
from ..classes import NodePosition

_EPSILON = 0.001
logger = logging.getLogger(__name__)


def node_positions_overlap(node_1: NodePosition, node_2: NodePosition) -> bool:
    return is_overlap(node_1.x, node_1.y, node_1.size, node_2.x, node_2.y, node_2.size)


def is_overlap(
    node_1_x: float,
    node_1_y: float,
    node_1_size: float,
    node_2_x: float,
    node_2_y: float,
    node_2_size: float
) -> bool:  # type: ignore
    min_dist = node_1_size + node_2_size
    if abs(node_2_x - node_1_x) > min_dist or abs(node_2_y - node_1_y) > min_dist:
        # shortcut to reduce the amount of math required, the distance calc is more expensive
        return False
    d = distance.euclidean([node_1_x, node_1_y], [node_2_x, node_2_y])
    return d <= node_1_size + node_2_size


def is_overlapping_any_node_and_index(node, new_x, new_y, nodes, start, end):
    overlapping = None
    idx = 0
    for idx, n in enumerate(nodes[start:end]):
        if n.node_id == node.node_id:
            # don't check against self
            continue
        if is_overlap(n.x, n.y, n.size, new_x, new_y, node.size):
            overlapping = n
            break
    return idx + start, overlapping


def is_overlapping_any_node_and_index_with_grid(
    node, new_x, new_y, nodes, start, end, size, grid
):
    overlapping = None
    idx = 0
    potentially_over_lapping = grid.get_potential_overlapping_nodes(new_x, new_y, size)
    for idx, n in enumerate(nodes[start:end]):
        if n not in potentially_over_lapping:
            continue
        if n.node_id == node.node_id:
            # don't check against self
            continue
        if is_overlap(n.x, n.y, n.size, new_x, new_y, node.size):
            overlapping = n
            break
    return idx + start, overlapping


def stats_nodes(nodes: List[_Node]) -> Tuple[float, float, float, float, float]:
    max_x = -math.inf
    min_x = math.inf
    max_y = -math.inf
    min_y = math.inf
    max_size = -1
    for node in nodes:
        max_x = max(node.x, max_x)
        max_y = max(node.y, max_y)
        min_x = min(node.x, min_x)
        min_y = min(node.y, min_y)
        max_size = max(node.size, max_size)
    return min_x, min_y, max_x, max_y, max_size


def total_area(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float
) -> float:
    return (max_x - min_x) * (max_y - min_y)


def move_point_on_line(a, b, ratio) -> float:
    npa = np.array(a)
    npb = np.array(b)
    d = npb - npa
    d_norm = normalize([d])
    new_point = b + d_norm[0] * ratio
    return new_point


def scale_graph(g, scale_factor):
    for _, n in g.items():
        n.x, n.y = move_point_on_line([0, 0], [n.x, n.y], scale_factor)
    return g


class _QuadNode:
    """
    Represents a node in a quad tree. Each node has a list of nodes that are represented here
    or in its children.
    Each node knows its own depth in the tree. Each node will have less than max_nodes_per_quad
    in the nodes list or it will have populated children.
    Each node also has a point back up to its parent.  The root node in a tree will have None for
    its parent attribute.
    Each node has an x and y that are the center point which is a weighted center.
    Each node has min_x, min_y, max_x_, max_y that defines the area of this node
    Each node also has four child nodes NW, NE, SE, and SW they represent the nodes that are
    that direction from the center.

    Each node has a property indicating if has been laid out.
    """

    max_ratio = 0.95

    def __init__(
        self,
        nodes: List[_Node],
        depth: int,
        max_nodes_per_quad: int,
        parent: Optional["_QuadNode"] = None
    ):
        self.NW = None
        self.NE = None
        self.SW = None
        self.SE = None
        self.x = 0.0
        self.y = 0.0
        self.x_range = 0.0
        self.y_range = 0.0
        self.circle_size = 0.0
        self.square_size = 0.0
        self.sq_ratio = 0.0
        self.cir_ratio = 0.0
        self.total_cells = 0
        self.min_x = 0.0
        self.max_x = 0.0
        self.min_y = 0.0
        self.max_y = 0.0
        self.max_size = 0.0

        self.nodes = nodes
        self.depth = depth
        self.max_nodes_per_quad = max_nodes_per_quad
        self.is_laid_out = False
        self.parent = parent

        self.find_center()
        self.push_to_kids()

        self.total_nodes_moved = 0
        self.not_first_choice = 0

    def __lt__(self, other):
        return self.x < other.x

    def child_list(self) -> List["_QuadNode"]:
        return [self.NW, self.NE, self.SW, self.SE]

    def get_total_cells(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        max_size: float
    ) -> int:
        side_size = max_size * 2.0
        self.x_range = max_x - min_x
        self.y_range = max_y - min_y
        number_of_x_cells = self.x_range // side_size
        number_of_y_cells = self.y_range // side_size
        total_cells = number_of_y_cells * number_of_x_cells
        return int(total_cells)

    def find_center(self) -> None:
        if len(self.nodes) <= 0:
            raise Exception("Invalid to create a quad node with zero nodes!")

        self.min_x, self.min_y, self.max_x, self.max_y, self.max_size = stats_nodes(
            self.nodes
        )
        self.circle_size = self.total_circle_size()
        self.square_size = self.total_square_size()
        tot_area = total_area(self.min_x, self.min_y, self.max_x, self.max_y)
        if tot_area == 0:
            tot_area = 0.001
        self.x, self.y = 0.0, 0.0
        self.total_cells = self.get_total_cells(
            self.min_x, self.min_y, self.max_x, self.max_y, self.max_size
        )
        if tot_area == 0:
            self.sq_ratio = 1.0
            self.cir_ratio = 1.0
        else:
            self.sq_ratio = self.square_size / tot_area
            self.cir_ratio = self.circle_size / tot_area

        # sum X and Y values then devide by number of nodes to get the average (x,y) or center
        for n in self.nodes:
            self.x += n.x
            self.y += n.y
        # need to fix x and y
        self.x = self.x / self.num_nodes()
        self.y = self.y / self.num_nodes()

    def push_to_kids(self) -> None:
        if len(self.nodes) <= self.max_nodes_per_quad:
            # if len == nodes then we already have the center and mass initialized because of
            # the calculation done in the constructor
            self.NW = None
            self.NE = None
            self.SW = None
            self.SE = None
            return
        nw_nodes, sw_nodes, ne_nodes, se_nodes = [], [], [], []
        for node in self.nodes:
            if node.y > self.y:
                if node.x > self.x:
                    ne_nodes.append(node)
                else:
                    nw_nodes.append(node)
            else:
                if node.x > self.x:
                    se_nodes.append(node)
                else:
                    sw_nodes.append(node)

        self.NW = (
            _QuadNode(nw_nodes, self.depth + 1, self.max_nodes_per_quad, self)
            if len(nw_nodes) > 0
            else None
        )
        self.NE = (
            _QuadNode(ne_nodes, self.depth + 1, self.max_nodes_per_quad, self)
            if len(ne_nodes) > 0
            else None
        )
        self.SW = (
            _QuadNode(sw_nodes, self.depth + 1, self.max_nodes_per_quad, self)
            if len(sw_nodes) > 0
            else None
        )
        self.SE = (
            _QuadNode(se_nodes, self.depth + 1, self.max_nodes_per_quad, self)
            if len(se_nodes) > 0
            else None
        )

        for quad in self.child_list():
            if quad:
                quad.push_to_kids()

    def num_children(self) -> int:
        total = 1
        for quad in self.child_list():
            if quad:
                total += quad.num_children()
        return total

    def num_nodes(self) -> int:
        return len(self.nodes)

    def total_circle_size(self) -> float:
        total_size = 0
        for n in self.nodes:
            total_size += n.size * n.size * math.pi
        return total_size

    def total_square_size(self) -> float:
        total_size = 0
        for n in self.nodes:
            total_size += n.size * n.size * 2 * 2
        return total_size

    def find_grid_cell_and_center(
        self,
        min_x: float,
        min_y: float,
        max_size: float,
        x: float,
        y: float
    ) -> Tuple[int, int, float, float]:
        # zero the coordinates
        side_size = max_size * 2.0
        zeroed_x = x - min_x
        zeroed_y = y - min_y
        x_cell = int(zeroed_x // side_size)
        y_cell = int(zeroed_y // side_size)
        return x_cell, y_cell, min_x + side_size * x_cell, min_y + side_size * y_cell

    def find_free_cell(
        self,
        cells: Set[Tuple[int, int]],
        x: float,
        y: float,
        num_x_cells: int,
        num_y_cells: int,
        min_x: float,
        min_y: float,
        max_size: float
    ) -> Tuple[float, float, float, float]:
        # zero the coordinates
        square_size = max_size * 2
        new_x, new_y = x, y
        distance_to_move = 1
        count = 0
        done = True
        while (new_x, new_y) in cells or not done:
            done = False
            # go right
            for new_x in range(new_x + 1, new_x + distance_to_move + 1):
                if (
                    not (new_x, new_y) in cells
                    and new_x < num_x_cells
                    and new_y < num_y_cells
                    and new_x >= 0
                    and new_y >= 0
                ):
                    # print('breaking-r')
                    done = True
                    break
            if done:
                break
            # go up
            for new_y in range(new_y + 1, new_y + distance_to_move + 1):
                if (
                    not (new_x, new_y) in cells
                    and new_x < num_x_cells
                    and new_y < num_y_cells
                    and new_x >= 0
                    and new_y >= 0
                ):
                    # print('breaking-u')
                    done = True
                    break
            distance_to_move += 1
            if done:
                break
            # go left
            for new_x in range(new_x - 1, new_x - distance_to_move - 1, -1):
                if (
                    not (new_x, new_y) in cells
                    and new_x < num_x_cells
                    and new_y < num_y_cells
                    and new_x >= 0
                    and new_y >= 0
                ):
                    # print('breaking-l')
                    done = True
                    break
            if done:
                break
            # go down
            for new_y in range(new_y - 1, new_y - distance_to_move - 1, -1):
                if (
                    not (new_x, new_y) in cells
                    and new_x < num_x_cells
                    and new_y < num_y_cells
                    and new_x >= 0
                    and new_y >= 0
                ):
                    done = True
                    break
            distance_to_move += 1
            if done:
                break
            # keep going
            count += 1

        return new_x, new_y, min_x + square_size * new_x, min_y + square_size * new_y

    def layout_node_list(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        max_size: float,
        node_list: List[_Node]
    ) -> int:
        """
        This method will layout the nodes in the current quad.  If there are more nodes
            than cells an Exception will be raised.

        :param min_x:
        :param min_y:
        :param max_x:
        :param max_y:
        :param max_size:
        :param node_list:
        :return:
        """
        self._mark_laid_out()
        nodes_by_size = sorted(node_list, key=lambda n: n.size, reverse=True)
        largest_size = nodes_by_size[0].size
        if largest_size != max_size:
            raise Exception(
                "This can not be!!! max: %g, largest: %g" % (max_size, largest_size)
            )
        num_nodes = len(node_list)

        side_size = max_size * 2.0
        x_range = max_x - min_x
        y_range = max_y - min_y
        number_of_x_cells = int(x_range // side_size)
        number_of_y_cells = int(y_range // side_size)
        total_cells = number_of_y_cells * number_of_x_cells

        if num_nodes > total_cells:
            raise Exception(
                f"Too many nodes per Cell for this quad! nodes: {num_nodes}, "
                f"cells: {total_cells}"
            )

        number_overlapping = 0
        for idx, node_to_move in enumerate(nodes_by_size):
            for placed_node in nodes_by_size[idx + 1 :]:
                if is_overlap(
                    node_to_move.x,
                    node_to_move.y,
                    node_to_move.size,
                    placed_node.x,
                    placed_node.y,
                    placed_node.size,
                ):
                    number_overlapping += 1
                    break
            if number_overlapping > 0:
                break
        if number_overlapping == 0:
            return 0

        cells = set()
        for idx, node_to_move in enumerate(nodes_by_size):
            # the first one does not need to move all the rest might need to move
            (
                cell_x,
                cell_y,
                cell_center_x,
                cell_center_y,
            ) = self.find_grid_cell_and_center(
                min_x, min_y, max_size, node_to_move.x, node_to_move.y
            )
            found = (cell_x, cell_y) in cells
            self.total_nodes_moved += 1
            if found:
                # need to find an open cell and then move there.
                cell_x, cell_y, cell_center_x, cell_center_y = self.find_free_cell(
                    cells,
                    cell_x,
                    cell_y,
                    number_of_x_cells,
                    number_of_y_cells,
                    min_x,
                    min_y,
                    max_size,
                )
                self.not_first_choice += 1
            # Just a test to see how many move
            cells.add((cell_x, cell_y))
            node_to_move.x = cell_center_x
            node_to_move.y = cell_center_y

        return number_overlapping

    def _mark_laid_out(self) -> None:
        """
        Recursively mark all children as being laid out already
        :return:
        """
        self.is_laid_out = True
        for qn in self.child_list():
            if qn is not None:
                qn._mark_laid_out()

    def get_new_bounds(self, min_x, min_y, max_x, max_y, max_size, nodes):
        xrange = max_x - min_x
        yrange = max_y - min_y
        side_size = 2 * max_size
        x_cells = xrange // side_size
        y_cells = yrange // side_size
        total_cells = y_cells * x_cells
        cells_needed = len(nodes)
        # print ('original_total cells %d, (%d, %d) needed: %d' %(total_cells, x_cells, y_cells, cells_needed))
        while total_cells < cells_needed:
            x_cells += 1
            y_cells += 1
            total_cells = y_cells * x_cells
            # print ('new_cells %d, (%d, %d) needed: %d' %(total_cells, x_cells, y_cells, cells_needed))
        new_xrange = x_cells * side_size + 2
        new_yrange = y_cells * side_size + 2
        # print("bounds", min_x, min_y, max_x, max_y, new_xrange, new_yrange, side_size, x_cells, y_cells, total_cells)
        expanded_min_x = min_x - (new_xrange - xrange) / 2
        expanded_min_y = min_y - (new_yrange - yrange) / 2
        expanded_max_x = max_x + (new_xrange - xrange) / 2
        expanded_max_y = max_y + (new_yrange - yrange) / 2

        ## now I have the number of cells we need to go back and expand the bounds
        return expanded_min_x, expanded_min_y, expanded_max_x, expanded_max_y

    def layout_quad(self):
        # print("layout_quad")
        num_skipped = 0
        num_nodes = len(self.nodes)
        if self.total_cells == 0:
            nodes_per_cell = math.inf
        else:
            nodes_per_cell = num_nodes / self.total_cells

        if self.is_laid_out:
            # print ("ALREADY LAID OUT!!, ratio: %g " %(nodes_per_cell))
            return num_skipped

        has_children = False
        for quad in self.child_list():
            if quad:
                has_children = True
        if not has_children:
            if num_nodes > self.total_cells:
                logger.info(
                    "We don't fit! going up one level depth: %d, cells: %d, nodes %d, ratio: %g, max_size: %g"
                    % (
                        self.depth,
                        self.total_cells,
                        num_nodes,
                        nodes_per_cell,
                        self.max_size,
                    )
                )
                parent = self.parent
                # if parent is not None:
                while parent is not None:
                    logger.info(
                        "parent: sq_ratio: %g, cir_ratio: %g, cells %d, nodes: %d current_level %d, max_size: %g"
                        % (
                            parent.sq_ratio,
                            parent.cir_ratio,
                            parent.total_cells,
                            len(parent.nodes),
                            parent.depth,
                            parent.max_size,
                        )
                    )
                    if len(parent.nodes) > parent.total_cells:
                        # go up one more level
                        logger.info(
                            "A Quad at level %d does not have enough space to layout its nodes"
                            % (parent.depth)
                        )
                        parent = parent.parent
                    else:
                        # min_x, min_y, max_x, max_y, max_size = stats_nodes(parent.nodes)
                        # for n in parent.nodes:
                        # 	n.color = '#FF0004'
                        overlapping = parent.layout_node_list(
                            parent.min_x,
                            parent.min_y,
                            parent.max_x,
                            parent.max_y,
                            parent.max_size,
                            parent.nodes,
                        )
                        parent._do_contraction()
                        break
                if parent is None:
                    # expand the canvas and try to lay it out.
                    root = self.get_top_quad_node()
                    (
                        expanded_min_x,
                        expanded_min_y,
                        expanded_max_x,
                        expanded_max_y,
                    ) = self.get_new_bounds(
                        root.min_x,
                        root.min_y,
                        root.max_x,
                        root.max_y,
                        root.max_size,
                        root.nodes,
                    )
                    overlapping = root.layout_node_list(
                        expanded_min_x,
                        expanded_min_y,
                        expanded_max_x,
                        expanded_max_y,
                        root.max_size,
                        root.nodes,
                    )
                    # Just fixed this no longer need to throw the exception.
                    self._do_contraction_with_given_nodes(root.nodes)
                    # raise Exception('This root level does not have enough space to layout this graph')
                # print ("quad too dense, nodes_per_cell: %g, nn: %d, level: %d" %(nodes_per_cell, len(self.nodes), self.depth))
                # print ("quad (%g, %g) (%g, %g) max_size %g, ss: %g, area: %g, ratio: %g" %(parent.min_x, parent.min_y, parent.max_x, parent.max_y, parent.max_size, square_size, tot_area, ratio))
                return 1
            else:
                # we are at the bottom and we can fit everyone in here.
                # print ("laying out quad (%g, %g) (%g, %g) max_size %g, ss: %g, area: %g, ratio: %g, nodes: %d, npc: %g" %(self.min_x, self.min_y, self.max_x, self.max_y, self.max_size, square_size, tot_area, ratio, len(self.nodes), nodes_per_cell))
                overlapping = self.layout_node_list(
                    self.min_x,
                    self.min_y,
                    self.max_x,
                    self.max_y,
                    self.max_size,
                    self.nodes,
                )
                # print ("Should Fit")
                self._do_contraction()
                # print ("jiggled nodes, overlapping %d" %(overlapping))
        else:
            for quad in self.child_list():
                if quad:
                    num_skipped += quad.layout_quad()
        return num_skipped

    def quad_stats(self):
        square_size = self.total_square_size()
        tot_area = total_area(self.min_x, self.min_y, self.max_x, self.max_y)
        if tot_area == 0:
            tot_area = 0.01
        num_quad_no_kids = 0
        num_quad_to_dense = 0
        num_quad_fits = 0
        total_nodes_moved = self.total_nodes_moved
        not_first_choice = self.not_first_choice
        lowest_level = math.inf
        max_nodes_in_grid = 0

        has_children = False
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                no_kids, too_dense, fits, tnm, nfc, low_level, mng = quad.quad_stats()
                num_quad_no_kids += no_kids
                num_quad_to_dense += too_dense
                num_quad_fits += fits
                total_nodes_moved += tnm
                not_first_choice += nfc
                has_children = True
                lowest_level = min(lowest_level, low_level)
                max_nodes_in_grid = max(max_nodes_in_grid, mng)

        if not has_children:
            num_quad_no_kids += 1
            if len(self.nodes) > self.total_cells:
                if self.total_cells == 0:
                    nodes_to_cells = math.inf
                else:
                    nodes_to_cells = len(self.nodes) / self.total_cells
                # print ("too dense, nodes/cells: %g, nn: %d, cells: %d, level: %d" %(nodes_to_cells, len(self.nodes), self.total_cells, self.depth))
                num_quad_to_dense = 1
                lowest_level = math.inf
                parent = self.parent
                while parent is not None:
                    max_nodes_in_grid = parent.num_nodes()
                    lowest_level = parent.depth
                    if parent.num_nodes() > parent.total_cells:
                        # doesn't fit, go up one more level
                        parent = parent.parent
                    else:
                        break
            else:
                # we are at the bottom and we can fit everyone in here.
                lowest_level = self.depth
                max_nodes_in_grid = self.num_nodes()
                num_quad_fits = 1
        return (
            num_quad_no_kids,
            num_quad_to_dense,
            num_quad_fits,
            total_nodes_moved,
            not_first_choice,
            lowest_level,
            max_nodes_in_grid,
        )

    def num_overlapping(self):
        has_children = False
        num_overlapping_nodes = 0
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                overlapping = quad.num_overlapping()
                num_overlapping_nodes += overlapping
                has_children = True

        if not has_children:
            for idx, node_a in enumerate(self.nodes):
                for node_b in self.nodes[:idx]:
                    if is_overlap(
                        node_a.x, node_a.y, node_a.size, node_b.x, node_b.y, node_b.size
                    ):
                        num_overlapping_nodes += 1
                        # print ("***Is OVERLAPPING!!!!, to_move: (%g,%g) %g, (%g,%g) %g" % (node_a.x, node_a.y, node_a.size, node_b.x, node_b.y, node_b.size))
                        break
        return num_overlapping_nodes

    def is_just_outside_box(self, minx, miny, maxx, maxy, maxsize, x, y, size):
        dist = maxsize + size
        if x > minx - dist and x < maxx + dist:
            if y <= maxy + dist and y > miny - dist:
                # Here we are inside the box with buffer around it. now we need to check we are in the area just outside
                # of the original box
                if y > maxy:
                    return True
                elif y < miny:
                    return True
                elif x > maxx:
                    return True
                elif x < minx:
                    return True
                else:
                    return False
        else:
            return False

    def get_top_quad_node(self):
        tmp = self.parent
        prev = self
        while tmp is not None:
            prev = tmp
            tmp = tmp.parent
        return prev

    def get_nodes_near_lines(self, all_nodes):
        nodes_just_outside = []
        for n in all_nodes:
            if self.is_just_outside_box(
                self.min_x,
                self.min_y,
                self.max_x,
                self.max_y,
                self.max_size,
                n.x,
                n.y,
                n.size,
            ):
                nodes_just_outside.append(n)
        return nodes_just_outside

    def num_overlapping_across_quads(self, all_nodes):
        has_children = False
        num_overlapping_nodes = 0
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                overlapping = quad.num_overlapping_across_quads(all_nodes)
                num_overlapping_nodes += overlapping
                has_children = True

        if not has_children:
            nodes_to_check = self.get_nodes_near_lines(all_nodes)
            for idx, node_a in enumerate(self.nodes):
                # the first one does not need to move all the rest might need to move
                for node_b in nodes_to_check:
                    if node_a.node_id == node_b.node_id:
                        continue
                    if is_overlap(
                        node_a.x, node_a.y, node_a.size, node_b.x, node_b.y, node_b.size
                    ):
                        num_overlapping_nodes += 1
                        # print ("***Is OVERLAPPING!!!! across quads, to_move: (%g,%g) %g, (%g,%g) %g" % (node_a.x, node_a.y, node_a.size, node_b.x, node_b.y, node_b.size))
                        break
        return num_overlapping_nodes

    def get_density_list(self):
        square_size = self.total_square_size()
        tot_area = total_area(self.min_x, self.min_y, self.max_x, self.max_y)
        if tot_area == 0:
            tot_area = 0.001
        ratio = square_size / tot_area

        retval = []
        has_children = False
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                retval += quad.get_density_list()
                has_children = True

        if not has_children:
            if self.total_cells == 0:
                nodes_to_cells = math.inf
            else:
                nodes_to_cells = len(self.nodes) / self.total_cells
            return [(nodes_to_cells, ratio, self.total_cells, self)]
        else:
            return retval

    def get_overlapping_node_list(self, node, new_x, new_y, nodes):
        overlapping_nodes = []
        for n in nodes:
            if n.node_id == node.node_id:
                # don't check against self
                continue
            if is_overlap(n.x, n.y, n.size, new_x, new_y, node.size):
                overlapping_nodes.append(n)
        return overlapping_nodes

    def is_overlapping_any_node(self, node, new_x, new_y, nodes):
        overlapping = None
        for n in nodes:
            if n.node_id == node.node_id:
                # don't check against self
                continue
            if is_overlap(n.x, n.y, n.size, new_x, new_y, node.size):
                overlapping = n
                break
        return overlapping

    def is_between(self, x, one_end, other_end):
        if x <= one_end:
            return x >= other_end
        else:
            return x <= other_end

    ### I wanted to add a little thank you to the webiste: https://www.calculator.net/triangle-calculator.html
    ### it helped me debug the issues I was having in the calculation of the overlaps.
    def _do_contraction(self):
        logger.info("contracting nodes:%d" % (len(self.nodes)))
        node_list = self.nodes
        nodes_by_size = sorted(node_list, key=lambda n: n.size, reverse=True)
        num_nodes = len(node_list)
        nodes_around_the_edge = self.get_nodes_near_lines(
            self.get_top_quad_node().nodes
        )
        num_nodes_around_edge = len(nodes_around_the_edge)

        cells = {}
        for idx, node_to_move in enumerate(nodes_by_size):
            # move to its original spot node_to_move.original_x, node_to_move.original_y
            # then move it toward where it is until it does not overlap with anything already placed.
            prev_x, prev_y = node_to_move.original_x, node_to_move.original_y
            new_x, new_y = node_to_move.x, node_to_move.y
            ov_idx = 0
            ov_idx, overlapping_node = is_overlapping_any_node_and_index(
                node_to_move,
                node_to_move.original_x,
                node_to_move.original_y,
                nodes_around_the_edge + nodes_by_size,
                ov_idx,
                idx + num_nodes_around_edge,
            )
            if overlapping_node is None:
                new_x, new_y = prev_x, prev_y

            if node_to_move.x == node_to_move.original_x:
                # this is needed just in case the min x node is overlapping.
                # then the orginal X is eual to the X where is moves and that give us a divide by zero
                # when calculating the slope
                # We wiggle it just a little bit to prevent an error
                node_to_move.x += _EPSILON

            # print ("contracting: %d, node_to_move %s, overlapping: %s" % (idx, str(node_to_move.to_list()), overlapping_node))
            while (
                overlapping_node is not None
                and node_to_move.x != node_to_move.original_x
            ):
                # slope doesn't change leave as original_xy
                slope_ca = (node_to_move.y - node_to_move.original_y) / (
                    node_to_move.x - node_to_move.original_x
                )
                if node_to_move.node_id == overlapping_node.node_id:
                    raise Exception(
                        "They should not be the same node!! %s" % (node_to_move.node_id)
                    )
                if node_to_move.original_x == new_x:
                    new_x += _EPSILON
                a = dist_original_to_over = distance.euclidean(
                    [node_to_move.original_x, node_to_move.original_y],
                    [overlapping_node.x, overlapping_node.y],
                )
                b = dist_from_original_to_new = distance.euclidean(
                    [node_to_move.original_x, node_to_move.original_y], [new_x, new_y]
                )
                c = dist_from_new_to_overlapping = distance.euclidean(
                    [new_x, new_y], [overlapping_node.x, overlapping_node.y]
                )
                node_to_move.color = "#FF0004"  # RED
                overlapping_node.color = "#F1FD00"  # Yellow
                # print ("not None, a: %g, b: %g, c: %g" %(a, b, c), node_to_move.node_id, node_to_move.size, overlapping_node.size)
                # print ("original(%g,%g), current(%g,%g), overlap(%g,%g)" %(node_to_move.original_x, node_to_move.original_y, new_x, new_y, overlapping_node.x, overlapping_node.y))
                angle_c = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                len_c_new = node_to_move.size + overlapping_node.size + _EPSILON
                angle_a_new = math.asin(a * math.sin(angle_c) / len_c_new)
                angle_b_new = 180 - math.degrees(angle_c) - math.degrees(angle_a_new)
                new_len_b = (
                    len_c_new * math.sin(math.radians(angle_b_new)) / math.sin(angle_c)
                )
                # print ("slope: %g, angle c: %g, new angle a: %g, newlenC: %g, new angle a: %g, new lenB %g" %(slope_ca, math.degrees(angle_c), math.degrees(angle_a_new), len_c_new, math.degrees(angle_a_new), new_len_b))
                x_new_plus = node_to_move.original_x + math.sqrt(
                    new_len_b ** 2 / (1 + slope_ca ** 2)
                )
                x_new_neg = node_to_move.original_x - math.sqrt(
                    new_len_b ** 2 / (1 + slope_ca ** 2)
                )
                x_plus_diff = x_new_plus - new_x
                x_neg_diff = x_new_neg - new_x
                # print ("both outsize, plus diff: %g, minus diff: %g" %(x_plus_diff, x_neg_diff))
                if abs(x_plus_diff) < abs(x_neg_diff):
                    prev_x, prev_y = new_x, new_y
                    new_x = x_new_plus
                    new_y = prev_y - slope_ca * prev_x + slope_ca * x_new_plus
                else:
                    prev_x, prev_y = new_x, new_y
                    new_x = x_new_neg
                    new_y = prev_y - slope_ca * prev_x + slope_ca * x_new_neg
                # print ("before: idx: %d, node: %s" %(ov_idx, str(overlapping_node.to_list()) ))
                ov_idx, overlapping_node = is_overlapping_any_node_and_index(
                    node_to_move,
                    new_x,
                    new_y,
                    nodes_around_the_edge + nodes_by_size,
                    0,
                    idx + num_nodes_around_edge,
                )
                # print ("after: idx: %d, node: %s" %(ov_idx, str(overlapping_node is not None) ))
            node_to_move.x = new_x
            node_to_move.y = new_y

            # if idx > 20: # only do a few in the first quad
            # 	break

        return

    def _do_contraction_with_given_nodes(self, node_list):
        logger.info("contracting nodes:%d" % (len(node_list)))
        nodes_by_size = sorted(node_list, key=lambda n: n.size, reverse=True)
        nodes_around_the_edge = []
        num_nodes_around_edge = len(nodes_around_the_edge)

        cells = {}
        for idx, node_to_move in enumerate(nodes_by_size):
            if idx % 100 == 0:
                logger.info(f"processing {idx}")
            # move to its original spot node_to_move.original_x, node_to_move.original_y
            # then move it toward where it is until it does not overlap with anything already placed.
            prev_x, prev_y = node_to_move.original_x, node_to_move.original_y
            new_x, new_y = node_to_move.x, node_to_move.y
            ov_idx = 0
            ov_idx, overlapping_node = is_overlapping_any_node_and_index(
                node_to_move,
                node_to_move.original_x,
                node_to_move.original_y,
                nodes_around_the_edge + nodes_by_size,
                ov_idx,
                idx + num_nodes_around_edge,
            )
            if overlapping_node is None:
                new_x, new_y = prev_x, prev_y

            if node_to_move.x == 0.0:
                # this is needed just in case the min x node is overlapping.
                # then the orginal X is equal to the X where is moves and that give us a divide by zero
                # when calculating the slope
                # We wiggle it just a little bit to make the math work
                node_to_move.x += _EPSILON

            # print ("contracting: %d, node_to_move %s, overlapping: %s" % (idx, str(node_to_move.to_list()), overlapping_node))
            while (
                overlapping_node is not None
                and node_to_move.x != node_to_move.original_x
            ):
                # slope doesn't change leave as original_xy
                slope_ca = (node_to_move.y - node_to_move.original_y) / (
                    node_to_move.x - node_to_move.original_x
                )
                if node_to_move.node_id == overlapping_node.node_id:
                    raise Exception(
                        "They should not be the same node!! %s" % (node_to_move.node_id)
                    )
                if node_to_move.original_x == new_x:
                    new_x += _EPSILON
                a = dist_original_to_over = distance.euclidean(
                    [node_to_move.original_x, node_to_move.original_y],
                    [overlapping_node.x, overlapping_node.y],
                )
                b = dist_from_original_to_new = distance.euclidean(
                    [node_to_move.original_x, node_to_move.original_y], [new_x, new_y]
                )
                c = dist_from_new_to_overlapping = distance.euclidean(
                    [new_x, new_y], [overlapping_node.x, overlapping_node.y]
                )
                # print ("not None, a: %g, b: %g, c: %g" %(a, b, c), node_to_move.node_id, node_to_move.size, overlapping_node.size)
                # print ("original(%g,%g), current(%g,%g), overlap(%g,%g)" %(node_to_move.original_x, node_to_move.original_y, new_x, new_y, overlapping_node.x, overlapping_node.y))
                denominator = 2 * a * b
                if 0 == denominator:
                    denominator = 0.00000001
                value = (a ** 2 + b ** 2 - c ** 2) / denominator
                if value > 1:
                    value = 0.999999
                elif value < -1:
                    value = -0.999999
                angle_c = math.acos(value)
                len_c_new = node_to_move.size + overlapping_node.size + _EPSILON
                angle_a_new = math.asin(a * math.sin(angle_c) / len_c_new)
                angle_b_new = 180 - math.degrees(angle_c) - math.degrees(angle_a_new)
                new_len_b = (
                    len_c_new * math.sin(math.radians(angle_b_new)) / math.sin(angle_c)
                )
                # print ("slope: %g, angle c: %g, new angle a: %g, newlenC: %g, new angle a: %g, new lenB %g" %(slope_ca, math.degrees(angle_c), math.degrees(angle_a_new), len_c_new, math.degrees(angle_a_new), new_len_b))
                x_new_plus = node_to_move.original_x + math.sqrt(
                    new_len_b ** 2 / (1 + slope_ca ** 2)
                )
                x_new_neg = node_to_move.original_x - math.sqrt(
                    new_len_b ** 2 / (1 + slope_ca ** 2)
                )
                x_plus_diff = x_new_plus - new_x
                x_neg_diff = x_new_neg - new_x
                # print ("both outsize, plus diff: %g, minus diff: %g" %(x_plus_diff, x_neg_diff))
                if abs(x_plus_diff) < abs(x_neg_diff):
                    prev_x, prev_y = new_x, new_y
                    new_x = x_new_plus
                    new_y = prev_y - slope_ca * prev_x + slope_ca * x_new_plus
                else:
                    prev_x, prev_y = new_x, new_y
                    new_x = x_new_neg
                    new_y = prev_y - slope_ca * prev_x + slope_ca * x_new_neg
                # print ("before: idx: %d, node: %s" %(ov_idx, str(overlapping_node.to_list()) ))
                ov_idx, overlapping_node = is_overlapping_any_node_and_index(
                    node_to_move,
                    new_x,
                    new_y,
                    nodes_around_the_edge + nodes_by_size,
                    0,
                    idx + num_nodes_around_edge,
                )
            # print ("after: idx: %d, node: %s" %(ov_idx, str(overlapping_node is not None) ))
            node_to_move.x = new_x
            node_to_move.y = new_y

        # if idx > 20: # only do a few in the first quad
        # 	break

        return

    def collect_nodes(self, all_nodes):
        has_children = False
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                quad.collect_nodes(all_nodes)
                has_children = True

        if not has_children:
            return all_nodes.extend(self.nodes)

    def boxes_by_level(self, boxes):
        has_children = False
        for quad in [self.NW, self.NE, self.SW, self.SE]:
            if quad:
                quad.boxes_by_level(boxes)
                has_children = True

        boxes.append((self.depth, self.min_x, self.min_y, self.max_x, self.max_y))
        return
