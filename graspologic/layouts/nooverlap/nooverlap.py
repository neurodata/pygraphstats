# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from typing import List

from ._node import _Node
from ._quad_tree import _QuadTree
from .. import NodePosition

logger = logging.getLogger(__name__)


def remove_overlaps(node_positions: List[NodePosition]) -> List[NodePosition]:
    start = time.time()
    logger.info("removing overlaps")
    local_nodes = [
        _Node(node.node_id, node.x, node.y, node.size, node.community)
        for node in node_positions
    ]
    qt = _QuadTree(local_nodes, 50)
    qt.layout_dense_first(first_color=None)
    stop = time.time()
    logger.info(f"removed overlap in {stop-start} seconds")

    new_positions = [
        NodePosition(
            node_id=node.node_id,
            x=node.x,
            y=node.y,
            size=node.size,
            community=node.community,
        )
        for node in local_nodes
    ]
    return new_positions
