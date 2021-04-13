# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any


class _Node:
    def __init__(
        self,
        node_id: Any,
        x: float,
        y: float,
        size: float,
        community: int = 9999999,
        color: str = ""
    ):
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)
        self.original_x = self.x
        self.original_y = self.y
        self.size = float(size)
        self.community = community
        self.color = color

    def reset_original_position(
        self,
        new_x: float,
        new_y: float
    ) -> None:
        self.original_x = self.x = new_x
        self.original_y = self.y = new_y

    def __eq__(self, other: Any) -> bool:
        return self.node_id == other.node_id  # type: ignore

    def __hash__(self) -> int:
        return hash(self.node_id)
