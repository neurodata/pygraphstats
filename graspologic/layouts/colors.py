# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import atexit
from itertools import cycle
import json
import math
import numpy as np
import os
from pathlib import Path
import pkg_resources
from sklearn.preprocessing import minmax_scale
from typing import Any, Dict, Optional, Tuple


__all__ = ["categorical_colors", "sequential_colors"]


def _load_thematic_json(path: Optional[str]) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    if path is not None and Path(path).is_file():
        colors_path = path
    else:
        atexit.register(pkg_resources.cleanup_resources)
        include_path = pkg_resources.resource_filename(__package__, "include")
        colors_path = os.path.join(include_path, "colors-100.json")

    with open(colors_path) as thematic_json_io:
        thematic_json = json.load(thematic_json_io)
    light = thematic_json["light"]
    dark = thematic_json["dark"]
    return light, dark


_CACHED_LIGHT, _CACHED_DARK = _load_thematic_json(None)


def _get_colors(light_background: bool, theme_path: Optional[str]) -> Dict[Any, Any]:
    (
        light,
        dark,
    ) = _CACHED_LIGHT, _CACHED_DARK if theme_path is None else _load_thematic_json(
        theme_path
    )
    return light if light_background else dark


def categorical_colors(
    partitions: Dict[Any, int],
    light_background: bool = True,
    theme_path: Optional[str] = None,
) -> Dict[Any, str]:
    """
    Generates a node -> color mapping based on the partitions provided.

    The partitions are ordered by population descending, and a series of perceptually
    balanced, complementary colors are chosen in sequence.

    If a theme_path is provided, it must contain a path to a json file generated by
    `Thematic <https://microsoft.github.io/thematic>`_, otherwise it will use the theme
    packaged with this library.

    Colors will be different when selecting for a light background vs. a dark
    background, using the principles defined by
    `Thematic <https://microsoft.github.io/thematic>`_.

    If more partitions than colors available (100) are selected, the colors will be
    cycled through again.

    Parameters
    ----------
    partitions : Dict[Any, int]
        A dictionary of node ids to partition ids.
    light_background : bool
        Default is ``True``. Colors selected for a light background will be slightly
        different in hue and saturation to complement a light or dark background.
    theme_path : Optional[str]
        A color scheme is provided with ``graspologic``, but if you wish to use your own
        you can generate one with `Thematic <https://microsoft.github.io/thematic>`_ and
        provide the path to it to override the bundled theme.

    Returns
    -------
    Dict[Any, str]
        Returns a dictionary of node id -> color based on the partitions provided.

    """
    color_scheme = _get_colors(light_background, theme_path)
    partition_populations = {}
    for node_id, partition in partitions.items():
        count = partition_populations.get(partition, 0) + 1
        partition_populations[partition] = count

    ordered_partitions = sorted(
        partition_populations.items(), key=lambda x: x[1], reverse=True
    )
    nominal_cycle = cycle(color_scheme["nominal"])
    colors_by_partitions = {}
    for index, item in enumerate(ordered_partitions):
        partition, _ = item
        color = next(nominal_cycle)
        colors_by_partitions[partition] = color

    colors_by_node = {
        node_id: colors_by_partitions[partition]
        for node_id, partition in partitions.items()
    }
    return colors_by_node


def sequential_colors(
    node_and_value: Dict[Any, float],
    light_background: bool = True,
    use_log_scale: bool = False,
    theme_path: Optional[str] = None,
) -> Dict[Any, str]:
    """
    Generates a node -> color mapping where a color is chosen for the value as it
    maps the value range into the sequential color space.

    If a theme_path is provided, it must contain a path to a json file generated by
    `Thematic <https://microsoft.github.io/thematic>`_, otherwise it will use the theme
    packaged with this library.

    Colors will be different when selecting for a light background vs. a dark
    background, using the principles defined by
    `Thematic <https://microsoft.github.io/thematic>`_.

    If more partitions than colors available (100) are selected, the colors will be
    cycled through again.

    Parameters
    ----------
    node_and_value : Dict[Any, float]
        A node to value mapping. The value is a single entry in a continuous range,
        which is then mapped into a sequential color space.
    light_background : bool
        Default is ``True``. Colors selected for a light background will be slightly
        different in hue and saturation to complement a light or dark background.
    use_log_scale : bool
        Default is ``False``.
    theme_path : Optional[str]
        A color scheme is provided with ``graspologic``, but if you wish to use your own
        you can generate one with `Thematic <https://microsoft.github.io/thematic>`_ and
        provide the path to it to override the bundled theme.

    Returns
    -------
    Dict[Any, str]
        Returns a dictionary of node id -> color based on the original value
        provided for the node as it relates to the total range of all values.

    """
    color_scheme = _get_colors(light_background, theme_path)
    color_list = color_scheme["sequential"]
    num_colors = len(color_list)

    keys, values = zip(*node_and_value.items())

    if use_log_scale:
        values = map(math.log, values)

    np_values = np.array(values).reshape(1, -1)
    new_values = minmax_scale(np_values, feature_range=(0, num_colors - 1), axis=1)
    node_colors = {}
    for key_index, node_id in enumerate(keys):
        index = int(new_values[0, key_index])

        color = color_list[index]
        node_colors[node_id] = color

    return node_colors
