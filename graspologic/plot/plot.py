# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import linalg
from sklearn.preprocessing import Binarizer
from sklearn.utils import check_array, check_consistent_length, check_X_y

from ..embed import selectSVD
from ..utils import import_graph, pass_to_ranks


def _check_common_inputs(
    figsize=None,
    height=None,
    title=None,
    context=None,
    font_scale=None,
    legend_name=None,
    title_pad=None,
    hier_label_fontsize=None,
):
    # Handle figsize
    if figsize is not None:
        if not isinstance(figsize, tuple):
            msg = "figsize must be a tuple, not {}.".format(type(figsize))
            raise TypeError(msg)

    # Handle heights
    if height is not None:
        if not isinstance(height, (int, float)):
            msg = "height must be an integer or float, not {}.".format(type(height))
            raise TypeError(msg)

    # Handle title
    if title is not None:
        if not isinstance(title, str):
            msg = "title must be a string, not {}.".format(type(title))
            raise TypeError(msg)

    # Handle context
    if context is not None:
        if not isinstance(context, str):
            msg = "context must be a string, not {}.".format(type(context))
            raise TypeError(msg)
        elif context not in ["paper", "notebook", "talk", "poster"]:
            msg = "context must be one of (paper, notebook, talk, poster), \
                not {}.".format(
                context
            )
            raise ValueError(msg)

    # Handle font_scale
    if font_scale is not None:
        if not isinstance(font_scale, (int, float)):
            msg = "font_scale must be an integer or float, not {}.".format(
                type(font_scale)
            )
            raise TypeError(msg)

    # Handle legend name
    if legend_name is not None:
        if not isinstance(legend_name, str):
            msg = "legend_name must be a string, not {}.".format(type(legend_name))
            raise TypeError(msg)

    if hier_label_fontsize is not None:
        if not isinstance(hier_label_fontsize, (int, float)):
            msg = "hier_label_fontsize must be a scalar, not {}.".format(
                type(legend_name)
            )
            raise TypeError(msg)

    if title_pad is not None:
        if not isinstance(title_pad, (int, float)):
            msg = "title_pad must be a scalar, not {}.".format(type(legend_name))
            raise TypeError(msg)


def _transform(arr, method):
    if method is not None:
        if method in ["log", "log10"]:
            # arr = np.log(arr, where=(arr > 0))
            # hacky, but np.log(arr, where=arr>0) is really buggy
            arr = arr.copy()
            if method == "log":
                arr[arr > 0] = np.log(arr[arr > 0])
            else:
                arr[arr > 0] = np.log10(arr[arr > 0])
        elif method in ["zero-boost", "simple-all", "simple-nonzero"]:
            arr = pass_to_ranks(arr, method=method)
        elif method == "binarize":
            transformer = Binarizer().fit(arr)
            arr = transformer.transform(arr)
        else:
            msg = f"Transform must be one of {{log, log10, binarize, zero-boost, \
            simple-all, simple-nonzero}}, not {method}."
            raise ValueError(msg)

    return arr


def _process_graphs(
    graphs, inner_hier_labels, outer_hier_labels, transform, sort_nodes
):
    """Handles transformation and sorting of graphs for plotting"""
    for g in graphs:
        check_consistent_length(g, inner_hier_labels, outer_hier_labels)

    graphs = [_transform(arr, transform) for arr in graphs]

    if inner_hier_labels is not None:
        inner_hier_labels = np.array(inner_hier_labels)
        if outer_hier_labels is None:
            outer_hier_labels = np.ones_like(inner_hier_labels)
        else:
            outer_hier_labels = np.array(outer_hier_labels)
    else:
        inner_hier_labels = np.ones(graphs[0].shape[0], dtype=int)
        outer_hier_labels = np.ones_like(inner_hier_labels)

    graphs = [
        _sort_graph(arr, inner_hier_labels, outer_hier_labels, sort_nodes)
        for arr in graphs
    ]
    return graphs


def heatmap(
    X,
    transform=None,
    figsize=(10, 10),
    title=None,
    context="talk",
    font_scale=1,
    xticklabels=False,
    yticklabels=False,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    center=0,
    cbar=True,
    inner_hier_labels=None,
    outer_hier_labels=None,
    hier_label_fontsize=30,
    ax=None,
    title_pad=None,
    sort_nodes=False,
    **kwargs,
):
    r"""
    Plots a graph as a color-encoded matrix.

    Nodes can be grouped by providing ``inner_hier_labels`` or both
    ``inner_hier_labels`` and ``outer_hier_labels``. Nodes can also
    be sorted by the degree from largest to smallest degree nodes.
    The nodes will be sorted within each group if labels are also
    provided.

    Read more in the `Heatmap: Visualizing a Graph Tutorial
    <https://microsoft.github.io/graspologic/tutorials/plotting/heatmaps.html>`_

    Parameters
    ----------
    X : nx.Graph or np.ndarray object
        Graph or numpy matrix to plot

    transform : None, or string {'log', 'log10', 'zero-boost', 'simple-all', 'simple-nonzero'}

        - 'log'
            Plots the natural log of all nonzero numbers
        - 'log10'
            Plots the base 10 log of all nonzero numbers
        - 'zero-boost'
            Pass to ranks method. preserves the edge weight for all 0s, but ranks
            the other edges as if the ranks of all 0 edges has been assigned.
        - 'simple-all'
            Pass to ranks method. Assigns ranks to all non-zero edges, settling
            ties using the average. Ranks are then scaled by
            :math:`\frac{rank(\text{non-zero edges})}{n^2 + 1}`
            where n is the number of nodes
        - 'simple-nonzero'
            Pass to ranks method. Same as simple-all, but ranks are scaled by
            :math:`\frac{rank(\text{non-zero edges})}{\text{# non-zero edges} + 1}`
        - 'binarize'
            Binarize input graph such that any edge weight greater than 0 becomes 1.

    figsize : tuple of integers, optional, default: (10, 10)
        Width, height in inches.

    title : str, optional, default: None
        Title of plot.

    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.

    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.

    xticklabels, yticklabels : bool or list, optional
        If list-like, plot these alternate labels as the ticklabels.

    cmap : str, list of colors, or matplotlib.colors.Colormap, default: 'RdBu_r'
        Valid matplotlib color map.

    vmin, vmax : floats, optional (default=None)
        Values to anchor the colormap, otherwise they are inferred from the data and
        other keyword arguments.

    center : float, default: 0
        The value at which to center the colormap

    cbar : bool, default: True
        Whether to draw a colorbar.

    inner_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes. If not None, will group the nodes
        according to these labels and plot the labels on the marginal

    outer_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes, ignored without ``inner_hier_labels``
        If not None, will plot these labels as the second level of a hierarchy on the
        marginals

    hier_label_fontsize : int
        Size (in points) of the text labels for the ``inner_hier_labels`` and
        ``outer_hier_labels``.

    ax : matplotlib Axes, optional
        Axes in which to draw the plot, otherwise will generate its own axes

    title_pad : int, float or None, optional (default=None)
        Custom padding to use for the distance of the title from the heatmap. Autoscales
        if ``None``

    sort_nodes : boolean, optional (default=False)
        Whether or not to sort the nodes of the graph by the sum of edge weights
        (degree for an unweighted graph). If ``inner_hier_labels`` is passed and
        ``sort_nodes`` is ``True``, will sort nodes this way within block.

    **kwargs : dict, optional
        additional plotting arguments passed to Seaborn's ``heatmap``
    """
    _check_common_inputs(
        figsize=figsize,
        title=title,
        context=context,
        font_scale=font_scale,
        hier_label_fontsize=hier_label_fontsize,
        title_pad=title_pad,
    )

    # Handle ticklabels
    if isinstance(xticklabels, list):
        if len(xticklabels) != X.shape[1]:
            msg = "xticklabels must have same length {}.".format(X.shape[1])
            raise ValueError(msg)
    elif not isinstance(xticklabels, bool):
        msg = "xticklabels must be a bool or a list, not {}".format(type(xticklabels))
        raise TypeError(msg)

    if isinstance(yticklabels, list):
        if len(yticklabels) != X.shape[0]:
            msg = "yticklabels must have same length {}.".format(X.shape[0])
            raise ValueError(msg)
    elif not isinstance(yticklabels, bool):
        msg = "yticklabels must be a bool or a list, not {}".format(type(yticklabels))
        raise TypeError(msg)

    # Handle cmap
    if not isinstance(cmap, (str, list, Colormap)):
        msg = "cmap must be a string, list of colors, or matplotlib.colors.Colormap,"
        msg += " not {}.".format(type(cmap))
        raise TypeError(msg)

    # Handle center
    if center is not None:
        if not isinstance(center, (int, float)):
            msg = "center must be a integer or float, not {}.".format(type(center))
            raise TypeError(msg)

    # Handle cbar
    if not isinstance(cbar, bool):
        msg = "cbar must be a bool, not {}.".format(type(center))
        raise TypeError(msg)

    arr = import_graph(X)

    arr = _process_graphs(
        [arr], inner_hier_labels, outer_hier_labels, transform, sort_nodes
    )[0]

    # Global plotting settings
    CBAR_KWS = dict(shrink=0.7)  # norm=colors.Normalize(vmin=0, vmax=1))

    with sns.plotting_context(context, font_scale=font_scale):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        plot = sns.heatmap(
            arr,
            cmap=cmap,
            square=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar_kws=CBAR_KWS,
            center=center,
            cbar=cbar,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        if title is not None:
            if title_pad is None:
                if inner_hier_labels is not None:
                    title_pad = 1.5 * font_scale + 1 * hier_label_fontsize + 30
                else:
                    title_pad = 1.5 * font_scale + 15
            plot.set_title(title, pad=title_pad)
        if inner_hier_labels is not None:
            if outer_hier_labels is not None:
                plot.set_yticklabels([])
                plot.set_xticklabels([])
                _plot_groups(
                    plot,
                    arr,
                    inner_hier_labels,
                    outer_hier_labels,
                    fontsize=hier_label_fontsize,
                )
            else:
                _plot_groups(plot, arr, inner_hier_labels, fontsize=hier_label_fontsize)
    return plot


def gridplot(
    X,
    labels=None,
    transform=None,
    height=10,
    title=None,
    context="talk",
    font_scale=1,
    alpha=0.7,
    sizes=(10, 200),
    palette="Set1",
    legend_name="Type",
    inner_hier_labels=None,
    outer_hier_labels=None,
    hier_label_fontsize=30,
    title_pad=None,
    sort_nodes=False,
):
    r"""
    Plots multiple graphs on top of each other with dots as edges.

    This function is useful for visualizing multiple graphs simultaneously.
    The size of the dots correspond to the edge weights of the graphs, and
    colors represent input graphs.

    Read more in the `Gridplot: Visualize Multiple Graphs Tutorial
    <https://microsoft.github.io/graspologic/tutorials/plotting/gridplot.html>`_

    Parameters
    ----------
    X : list of nx.Graph or np.ndarray object
        List of nx.Graph or numpy arrays to plot
    labels : list of str
        List of strings, which are labels for each element in X.
        ``len(X) == len(labels)``.
    transform : None, or string {'log', 'log10', 'zero-boost', 'simple-all', 'simple-nonzero'}

        - 'log'
            Plots the natural log of all nonzero numbers
        - 'log10'
            Plots the base 10 log of all nonzero numbers
        - 'zero-boost'
            Pass to ranks method. preserves the edge weight for all 0s, but ranks
            the other edges as if the ranks of all 0 edges has been assigned.
        - 'simple-all'
            Pass to ranks method. Assigns ranks to all non-zero edges, settling
            ties using the average. Ranks are then scaled by
            :math:`\frac{rank(\text{non-zero edges})}{n^2 + 1}`
            where n is the number of nodes
        - 'simple-nonzero'
            Pass to ranks method. Same as simple-all, but ranks are scaled by
            :math:`\frac{rank(\text{non-zero edges})}{\text{# non-zero edges} + 1}`
        - 'binarize'
            Binarize input graph such that any edge weight greater than 0 becomes 1.
    height : int, optional, default: 10
        Height of figure in inches.
    title : str, optional, default: None
        Title of plot.
    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.
    palette : str, dict, optional, default: 'Set1'
        Set of colors for mapping the ``hue`` variable. If a dict, keys should
        be values in the ``hue`` variable.
        For acceptable string arguments, see the palette options at
        :doc:`Choosing Colormaps in Matplotlib <tutorials/colors/colormaps>`.
    alpha : float [0, 1], default : 0.7
        Alpha value of plotted gridplot points
    sizes : length 2 tuple, default: (10, 200)
        Min and max size to plot edge weights
    legend_name : string, default: 'Type'
        Name to plot above the legend
    inner_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes. If not None, will group the nodes
        according to these labels and plot the labels on the marginal
    outer_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes, ignored without ``inner_hier_labels``
        If not None, will plot these labels as the second level of a hierarchy on the
        marginals
    hier_label_fontsize : int
        Size (in points) of the text labels for the ``inner_hier_labels`` and
        ``outer_hier_labels``.
    title_pad : int, float or None, optional (default=None)
        Custom padding to use for the distance of the title from the heatmap. Autoscales
        if ``None``
    sort_nodes : boolean, optional (default=False)
        Whether or not to sort the nodes of the graph by the sum of edge weights
        (degree for an unweighted graph). If ``inner_hier_labels`` is passed and
        ``sort_nodes`` is ``True``, will sort nodes this way within block.
    """
    _check_common_inputs(
        height=height,
        title=title,
        context=context,
        font_scale=font_scale,
        hier_label_fontsize=hier_label_fontsize,
        title_pad=title_pad,
    )

    if isinstance(X, list):
        graphs = [import_graph(x) for x in X]
    else:
        msg = "X must be a list, not {}.".format(type(X))
        raise TypeError(msg)

    if labels is None:
        labels = np.arange(len(X))

    check_consistent_length(X, labels)

    graphs = _process_graphs(
        X, inner_hier_labels, outer_hier_labels, transform, sort_nodes
    )

    if isinstance(palette, str):
        palette = sns.color_palette(palette, desat=0.75, n_colors=len(labels))

    dfs = []
    for idx, graph in enumerate(graphs):
        rdx, cdx = np.where(graph > 0)
        weights = graph[(rdx, cdx)]
        df = pd.DataFrame(
            np.vstack([rdx + 0.5, cdx + 0.5, weights]).T,
            columns=["rdx", "cdx", "Weights"],
        )
        df[legend_name] = [labels[idx]] * len(cdx)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    with sns.plotting_context(context, font_scale=font_scale):
        sns.set_style("white")
        plot = sns.relplot(
            data=df,
            x="cdx",
            y="rdx",
            hue=legend_name,
            size="Weights",
            sizes=sizes,
            alpha=alpha,
            palette=palette,
            height=height,
            facet_kws={
                "sharex": True,
                "sharey": True,
                "xlim": (0, graph.shape[0] + 1),
                "ylim": (0, graph.shape[0] + 1),
            },
        )
        plot.ax.axis("off")
        plot.ax.invert_yaxis()
        if title is not None:
            if title_pad is None:
                if inner_hier_labels is not None:
                    title_pad = 1.5 * font_scale + 1 * hier_label_fontsize + 30
                else:
                    title_pad = 1.5 * font_scale + 15
            plt.title(title, pad=title_pad)
    if inner_hier_labels is not None:
        if outer_hier_labels is not None:
            _plot_groups(
                plot.ax,
                graphs[0],
                inner_hier_labels,
                outer_hier_labels,
                fontsize=hier_label_fontsize,
            )
        else:
            _plot_groups(
                plot.ax, graphs[0], inner_hier_labels, fontsize=hier_label_fontsize
            )
    return plot


def pairplot(
    X,
    labels=None,
    col_names=None,
    title=None,
    legend_name=None,
    variables=None,
    height=2.5,
    context="talk",
    font_scale=1,
    palette="Set1",
    alpha=0.7,
    size=50,
    marker=".",
    diag_kind="auto",
):
    r"""
    Plot pairwise relationships in a dataset.

    By default, this function will create a grid of axes such that each dimension
    in data will by shared in the y-axis across a single row and in the x-axis
    across a single column.

    The off-diagonal axes show the pairwise relationships displayed as scatterplot.
    The diagonal axes show the univariate distribution of the data for that
    dimension displayed as either a histogram or kernel density estimates (KDEs).

    Read more in the `Pairplot: Visualizing High Dimensional Data Tutorial
    <https://microsoft.github.io/graspologic/tutorials/plotting/pairplot.html>`_

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    labels : array-like or list, shape (n_samples), optional
        Labels that correspond to each sample in ``X``.
    col_names : array-like or list, shape (n_features), optional
        Names or labels for each feature in ``X``. If not provided, the default
        will be `Dimension 1, Dimension 2, etc`.
    title : str, optional, default: None
        Title of plot.
    legend_name : str, optional, default: None
        Title of the legend.
    variables : list of variable names, optional
        Variables to plot based on col_names, otherwise use every column with
        a numeric datatype.
    height : int, optional, default: 10
        Height of figure in inches.
    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.
    palette : str, dict, optional, default: 'Set1'
        Set of colors for mapping the ``hue`` variable. If a dict, keys should
        be values in the ``hue`` variable.
        For acceptable string arguments, see the palette options at
        :doc:`Choosing Colormaps in Matplotlib <tutorials/colors/colormaps>`.
    alpha : float, optional, default: 0.7
        Opacity value of plotter markers between 0 and 1
    size : float or int, optional, default: 50
        Size of plotted markers.
    marker : string, optional, default: '.'
        Matplotlib marker specifier, see the marker options at
        :doc:`Matplotlib style marker specification <api/markers_api>`
    """
    _check_common_inputs(
        height=height,
        title=title,
        context=context,
        font_scale=font_scale,
        legend_name=legend_name,
    )

    # Handle X
    if not isinstance(X, (list, np.ndarray)):
        msg = "X must be array-like, not {}.".format(type(X))
        raise TypeError(msg)

    # Handle Y
    if labels is not None:
        if not isinstance(labels, (list, np.ndarray)):
            msg = "Y must be array-like or list, not {}.".format(type(labels))
            raise TypeError(msg)
        elif X.shape[0] != len(labels):
            msg = "Expected length {}, but got length {} instead for Y.".format(
                X.shape[0], len(labels)
            )
            raise ValueError(msg)

    # Handle col_names
    if col_names is None:
        col_names = ["Dimension {}".format(i) for i in range(1, X.shape[1] + 1)]
    elif not isinstance(col_names, list):
        msg = "col_names must be a list, not {}.".format(type(col_names))
        raise TypeError(msg)
    elif X.shape[1] != len(col_names):
        msg = "Expected length {}, but got length {} instead for col_names.".format(
            X.shape[1], len(col_names)
        )
        raise ValueError(msg)

    # Handle variables
    if variables is not None:
        if len(variables) > len(col_names):
            msg = "variables cannot contain more elements than col_names."
            raise ValueError(msg)
        else:
            for v in variables:
                if v not in col_names:
                    msg = "{} is not a valid key.".format(v)
                    raise KeyError(msg)
    else:
        variables = col_names

    df = pd.DataFrame(X, columns=col_names)
    if labels is not None:
        if legend_name is None:
            legend_name = "Type"
        df_labels = pd.DataFrame(labels, columns=[legend_name])
        df = pd.concat([df_labels, df], axis=1)

        names, counts = np.unique(labels, return_counts=True)
        if counts.min() < 2:
            diag_kind = "hist"
    plot_kws = dict(
        alpha=alpha,
        s=size,
        # edgecolor=None, # could add this latter
        linewidth=0,
        marker=marker,
    )
    with sns.plotting_context(context=context, font_scale=font_scale):
        if labels is not None:
            pairs = sns.pairplot(
                df,
                hue=legend_name,
                vars=variables,
                height=height,
                palette=palette,
                diag_kind=diag_kind,
                plot_kws=plot_kws,
            )
        else:
            pairs = sns.pairplot(
                df,
                vars=variables,
                height=height,
                palette=palette,
                diag_kind=diag_kind,
                plot_kws=plot_kws,
            )
        pairs.set(xticks=[], yticks=[])
        pairs.fig.subplots_adjust(top=0.945)
        pairs.fig.suptitle(title)

    return pairs


def _plot_ellipse_and_data(
    data, X, j, k, means, covariances, ax, label_palette, cluster_palette, alpha
):
    r"""
    plot_ellipse makes a scatter plot from the two dimensions j,k where j
    corresponds to x-axis
    and k corresponds to the y-axis onto the axis that is ax. plot_ellipse then
    applies a gmm ellipse onto the scatterplot
    using the data from Y_(which is stored in data["clusters"]),
    means, covariances.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    j : int
        column index of feature of interest from X which will be the x-axis data.
    k : int
        column index of feature of interest from X which will be the y-axis data.
    means : array-like, shape (n_components, n_features)
        Estimated means from gmm
    covariances : array-like, shape (with 'full') (n_components, n_features, n_features)
        estimated covariances from gmm
    ax : axis object
        The location where plot_ellipse will plot
    label_palette : dict, optional, default: dictionary using 'Set1'
    cluster_palette : dict, optional, default: dictionary using 'Set1'
    alpha : float, optional, default: 0.7
        Opacity value of plotter markers between 0 and 1

    References
    ----------
    .. [1]  https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py.
    """
    sns.scatterplot(
        data=data, x=X[:, j], y=X[:, k], ax=ax, hue="labels", palette=label_palette
    )
    for i, (mean, covar) in enumerate(zip(means, covariances)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(
            [mean[j], mean[k]],
            v[0],
            v[1],
            180.0 + angle,
            color=cluster_palette[i],
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(alpha)
        ax.add_artist(ell)
        # removes tick marks from off diagonal graphs
        ax.set(xticks=[], yticks=[], xlabel=k, ylabel=k)
        ax.legend().remove()


def pairplot_with_gmm(
    X,
    gmm,
    labels=None,
    cluster_palette="Set1",
    label_palette="Set1",
    title=None,
    legend_name=None,
    context="talk",
    font_scale=1,
    alpha=0.7,
    figsize=(12, 12),
    histplot_kws={},
):
    r"""
    Plot pairwise relationships in a dataset, also showing a clustering predicted by
    a Gaussian mixture model.

    By default, this function will create a grid of axes such that each dimension
    in data will by shared in the y-axis across a single row and in the x-axis
    across a single column.

    The off-diagonal axes show the pairwise relationships displayed as scatterplot.
    The diagonal axes show the univariate distribution of the data for that
    dimension displayed as either a histogram or kernel density estimates (KDEs).

    Read more in the `Pairplot with GMM: Visualizing High Dimensional Data and
    Clustering Tutorial
    <https://microsoft.github.io/graspologic/tutorials/plotting/pairplot_with_gmm.html>`_

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    gmm: GaussianMixture object
        A fit :class:`sklearn.mixture.GaussianMixture` object.
        Gaussian mixture models (GMMs) are probabilistic models for representing data
        based on normally distributed subpopulations, GMM clusters each data point into
        a corresponding subpopulation.
    labels : array-like or list, shape (n_samples), optional
        Labels that correspond to each sample in ``X``.
        If labels are not passed in then labels are predicted by ``gmm``.
    label_palette : str or dict, optional, default: 'Set1'
        Palette used to color points if ``labels`` are passed in.
    cluster_palette : str or dict, optional, default: 'Set1'
        Palette used to color GMM ellipses (and points if no ``labels`` are passed).
    title : string, default: ""
        Title of the plot.
    legend_name : string, default: None
        Name to put above the legend.
        If ``None``, will be "Cluster" if no custom ``labels`` are passed, and ""
        otherwise.
    context :  None, or one of {talk (default), paper, notebook, poster}
        Seaborn plotting context
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.
    alpha : float, optional, default: 0.7
        Opacity value of plotter markers between 0 and 1
    figsize : tuple
        The size of the 2d subplots configuration
    histplot_kws : dict, default: {}
        Keyword arguments passed down to :func:`seaborn.histplot`

    Returns
    -------
    fig : matplotlib Figure
    axes : np.ndarray
        Array of matplotlib Axes

    See Also
    --------
    graspologic.plot.pairplot
    graspologic.cluster.AutoGMMCluster
    sklearn.mixture.GaussianMixture
    """
    # Handle X and labels
    if labels is not None:
        check_X_y(X, labels)
        # if custom labels pass sets default
        if legend_name is None:
            legend_name = ""
    else:
        # sets default if no custom labels passed
        legend_name = "Cluster"
    # Handle gmm
    if gmm is None:
        msg = "You must input a sklearn.mixture.GaussianMixture"
        raise NameError(msg)
    Y_, means, covariances = gmm.predict(X), gmm.means_, gmm.covariances_
    data = pd.DataFrame(data=X)
    n_components = gmm.n_components

    # reformat covariances in preparation for ellipse plotting
    if gmm.covariance_type == "tied":
        covariances = np.repeat(
            gmm.covariances_[np.newaxis, :, :], n_components, axis=0
        )
    elif gmm.covariance_type == "diag":
        covariances = np.array(
            [np.diag(gmm.covariances_[i]) for i in range(n_components)]
        )
    elif gmm.covariance_type == "spherical":
        covariances = np.array(
            [
                np.diag(np.repeat(gmm.covariances_[i], X.shape[1]))
                for i in range(n_components)
            ]
        )

    # setting up the data DataFrame
    if labels is None:
        lab_names = [i for i in range(n_components)]
        data["labels"] = np.asarray([lab_names[Y_[i]] for i in range(Y_.shape[0])])
    else:
        data["labels"] = labels
    data["clusters"] = Y_
    # labels are given we must check whether input is correct
    if labels is not None:
        if isinstance(label_palette, str):
            colors = sns.color_palette(label_palette, n_components)
            label_palette = dict(zip(np.unique(np.asarray(labels)), colors))
        elif not isinstance(label_palette, dict):
            msg = "When giving labels must supply palette in string or dictionary"
            raise ValueError(msg)
        if isinstance(cluster_palette, str):
            colors = sns.color_palette(cluster_palette, n_components)
            cluster_palette = dict(zip(np.unique(Y_), colors))
        elif not isinstance(label_palette, dict):
            msg = "When giving labels must supply palette in string or dictionary"
            raise ValueError(msg)
    else:
        # no labels given we go to default.
        colors = sns.color_palette(cluster_palette, n_components)
        labels = np.unique(Y_)
        cluster_palette = dict(zip(np.unique(labels), colors))
        label_palette = dict(zip(np.unique(np.asarray(labels)), colors))

    with sns.plotting_context(context=context, font_scale=font_scale):
        dimensions = X.shape[1]
        # we only want 1 scatter plot for 2 features
        if X.shape[1] == 2:
            dimensions = 1
        fig, axes = plt.subplots(dimensions, dimensions, figsize=figsize, squeeze=False)
        # this will allow for uniform iteration whether axes was 2d or 1d
        axes = axes.flatten()
        for i in range(dimensions):
            for j in range(dimensions):
                if i == j and X.shape[1] > 2:
                    # take care of the histplot on diagonal
                    for t, lab in zip([i for i in range(X.shape[1])], label_palette):
                        sns.histplot(
                            X[Y_ == t, i],
                            ax=axes[dimensions * i + j],
                            color=label_palette[lab],
                            **histplot_kws,
                        )
                    # this removes the tick marks from the histplot
                    axes[dimensions * i + j].set_xticks([])
                    axes[dimensions * i + j].set_yticks([])
                else:
                    # take care off off-diagonal scatterplots
                    dim1, dim2 = j, i
                    # with only a scatter plot we must make sure we plot
                    # the first and second feature of X
                    if X.shape[1] == 2:
                        dim1, dim2 = 0, 1
                    _plot_ellipse_and_data(
                        data,
                        X,
                        dim1,
                        dim2,
                        means,
                        covariances,
                        axes[dimensions * i + j],
                        label_palette,
                        cluster_palette,
                        alpha=alpha,
                    )
        # formatting
        if title:
            plt.suptitle(title)
        for i in range(dimensions):
            for j in range(dimensions):
                if X.shape[1] == 2:
                    axes[dimensions * i + j].set_ylabel("Dimension " + str(1))
                    axes[dimensions * i + j].set_xlabel("Dimension " + str(2))
                else:
                    axes[dimensions * i + j].set_ylabel("Dimension " + str(i + 1))
                    axes[dimensions * i + j].set_xlabel("Dimension " + str(j + 1))

        for ax in axes.flat:
            ax.label_outer()
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
        # set up the legend correctly by only getting handles(colored dot)
        # and label corresponding to unique pairs
        if X.shape[1] == 2:
            handles, labels = axes[0].get_legend_handles_labels()
        else:
            handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center right",
            title=legend_name,
        )
        # allows for the legend to not overlap with plots while also keeping
        # legend in frame
        fig.subplots_adjust(right=0.85)
        return fig, axes


def _distplot(
    data,
    labels=None,
    direction="out",
    title="",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    palette="Set1",
    xlabel="",
    ylabel="Density",
):

    plt.figure(figsize=figsize)
    ax = plt.gca()
    palette = sns.color_palette(palette)
    plt_kws = {"cumulative": True}
    with sns.plotting_context(context=context, font_scale=font_scale):
        if labels is not None:
            categories, counts = np.unique(labels, return_counts=True)
            for i, cat in enumerate(categories):
                cat_data = data[np.where(labels == cat)]
                if counts[i] > 1 and cat_data.min() != cat_data.max():
                    x = np.sort(cat_data)
                    y = np.arange(len(x)) / float(len(x))
                    plt.plot(x, y, label=cat, color=palette[i])
                else:
                    ax.axvline(cat_data[0], label=cat, color=palette[i])
            plt.legend()
        else:
            if data.min() != data.max():
                sns.histplot(data, hist=False, kde_kws=plt_kws)
            else:
                ax.axvline(data[0])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return ax


def degreeplot(
    X,
    labels=None,
    direction="out",
    title="Degree plot",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    palette="Set1",
):
    r"""
    Plots the distribution of node degrees for the input graph.
    Allows for sets of node labels, will plot a distribution for each
    node category.

    Parameters
    ----------
    X : np.ndarray (2D)
        input graph
    labels : 1d np.ndarray or list, same length as dimensions of ``X``
        Labels for different categories of graph nodes
    direction : string, ('out', 'in')
        Whether to plot out degree or in degree for a directed graph
    title : string, default : 'Degree plot'
        Plot title
    context :  None, or one of {talk (default), paper, notebook, poster}
        Seaborn plotting context
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.
    palette : str, dict, optional, default: 'Set1'
        Set of colors for mapping the ``hue`` variable. If a dict, keys should
        be values in the ``hue`` variable.
        For acceptable string arguments, see the palette options at
        :doc:`Choosing Colormaps in Matplotlib <tutorials/colors/colormaps>`.
    figsize : tuple of length 2, default (10, 5)
        Size of the figure (width, height)

    Returns
    -------
    ax : matplotlib axis object
        Output plot
    """
    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale
    )
    check_array(X)
    if direction == "out":
        axis = 0
        check_consistent_length((X, labels))
    elif direction == "in":
        axis = 1
        check_consistent_length((X.T, labels))
    else:
        raise ValueError('direction must be either "out" or "in"')
    degrees = np.count_nonzero(X, axis=axis)
    ax = _distplot(
        degrees,
        labels=labels,
        title=title,
        context=context,
        font_scale=font_scale,
        figsize=figsize,
        palette=palette,
        xlabel="Node degree",
    )
    return ax


def edgeplot(
    X,
    labels=None,
    nonzero=False,
    title="Edge plot",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    palette="Set1",
):
    r"""
    Plots the distribution of edge weights for the input graph.
    Allows for sets of node labels, will plot edge weight distribution
    for each node category.

    Parameters
    ----------
    X : np.ndarray (2D)
        Input graph
    labels : 1d np.ndarray or list, same length as dimensions of ``X``
        Labels for different categories of graph nodes
    nonzero : boolean, default: False
        Whether to restrict the edgeplot to only the non-zero edges
    title : string, default : 'Edge plot'
        Plot title
    context :  None, or one of {talk (default), paper, notebook, poster}
        Seaborn plotting context
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.
    palette : str, dict, optional, default: 'Set1'
        Set of colors for mapping the ``hue`` variable. If a dict, keys should
        be values in the ``hue`` variable.
        For acceptable string arguments, see the palette options at
        :doc:`Choosing Colormaps in Matplotlib <tutorials/colors/colormaps>`.
    figsize : tuple of length 2, default (10, 5)
        Size of the figure (width, height)

    Returns
    -------
    ax : matplotlib axis object
        Output plot
    """
    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale
    )
    check_array(X)
    check_consistent_length((X, labels))
    edges = X.ravel()
    labels = np.tile(labels, (1, X.shape[1]))
    labels = labels.ravel()
    if nonzero:
        labels = labels[edges != 0]
        edges = edges[edges != 0]
    ax = _distplot(
        edges,
        labels=labels,
        title=title,
        context=context,
        font_scale=font_scale,
        figsize=figsize,
        palette=palette,
        xlabel="Edge weight",
    )
    return ax


def screeplot(
    X,
    title="Scree plot",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    cumulative=True,
    show_first=None,
):
    r"""
    Plots the distribution of singular values for a matrix, either showing the
    raw distribution or an empirical CDF (depending on ``cumulative``)

    Parameters
    ----------
    X : np.ndarray (2D)
        Input matrix
    title : string, default : 'Scree plot'
        Plot title
    context :  None, or one of {talk (default), paper, notebook, poster}
        Seaborn plotting context
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.
    figsize : tuple of length 2, default (10, 5)
        Size of the figure (width, height)
    cumulative : boolean, default: True
        Whether or not to plot a cumulative cdf of singular values
    show_first : int or None, default: None
        Whether to restrict the plot to the first ``show_first`` components

    Returns
    -------
    ax : matplotlib axis object
        Output plot
    """
    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale
    )
    check_array(X)
    if show_first is not None:
        if not isinstance(show_first, int):
            msg = "show_first must be an int"
            raise TypeError(msg)
    if not isinstance(cumulative, bool):
        msg = "cumulative must be a boolean"
        raise TypeError(msg)
    _, D, _ = selectSVD(X, n_components=X.shape[1], algorithm="full")
    D /= D.sum()
    if cumulative:
        y = np.cumsum(D[:show_first])
    else:
        y = D[:show_first]
    _ = plt.figure(figsize=figsize)
    ax = plt.gca()
    xlabel = "Component"
    ylabel = "Variance explained"
    with sns.plotting_context(context=context, font_scale=font_scale):
        plt.plot(y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    return ax


def _sort_inds(graph, inner_labels, outer_labels, sort_nodes):
    sort_df = pd.DataFrame(columns=("inner_labels", "outer_labels"))
    sort_df["inner_labels"] = inner_labels
    sort_df["outer_labels"] = outer_labels

    # get frequencies of the different labels so we can sort by them
    inner_label_counts = _get_freq_vec(inner_labels)
    outer_label_counts = _get_freq_vec(outer_labels)

    # inverse counts so we can sort largest to smallest
    # would rather do it this way so can still sort alphabetical for ties
    sort_df["inner_counts"] = len(inner_labels) - inner_label_counts
    sort_df["outer_counts"] = len(outer_labels) - outer_label_counts

    # get node edge sums (not exactly degrees if weighted)
    node_edgesums = graph.sum(axis=1) + graph.sum(axis=0)
    sort_df["node_edgesums"] = node_edgesums.max() - node_edgesums

    if sort_nodes:
        by = [
            "outer_counts",
            "outer_labels",
            "inner_counts",
            "inner_labels",
            "node_edgesums",
        ]
    else:
        by = ["outer_counts", "outer_labels", "inner_counts", "inner_labels"]
    sort_df.sort_values(by=by, kind="mergesort", inplace=True)

    sorted_inds = sort_df.index.values
    return sorted_inds


def _sort_graph(graph, inner_labels, outer_labels, sort_nodes):
    inds = _sort_inds(graph, inner_labels, outer_labels, sort_nodes)
    graph = graph[inds, :][:, inds]
    return graph


def _get_freqs(inner_labels, outer_labels=None):
    # use this because unique would give alphabetical
    _, outer_freq = _unique_like(outer_labels)
    outer_freq_cumsum = np.hstack((0, outer_freq.cumsum()))

    # for each group of outer labels, calculate the boundaries of the inner labels
    inner_freq = np.array([])
    for i in range(outer_freq.size):
        start_ind = outer_freq_cumsum[i]
        stop_ind = outer_freq_cumsum[i + 1]
        _, temp_freq = _unique_like(inner_labels[start_ind:stop_ind])
        inner_freq = np.hstack([inner_freq, temp_freq])
    inner_freq_cumsum = np.hstack((0, inner_freq.cumsum()))

    return inner_freq, inner_freq_cumsum, outer_freq, outer_freq_cumsum


def _get_freq_vec(vals):
    # give each set of labels a vector corresponding to its frequency
    _, inv, counts = np.unique(vals, return_counts=True, return_inverse=True)
    count_vec = counts[inv]
    return count_vec


def _unique_like(vals):
    # gives output like
    uniques, inds, counts = np.unique(vals, return_index=True, return_counts=True)
    inds_sort = np.argsort(inds)
    uniques = uniques[inds_sort]
    counts = counts[inds_sort]
    return uniques, counts


# assume that the graph has already been plotted in sorted form
def _plot_groups(ax, graph, inner_labels, outer_labels=None, fontsize=30):
    inner_labels = np.array(inner_labels)
    plot_outer = True
    if outer_labels is None:
        outer_labels = np.ones_like(inner_labels)
        plot_outer = False

    sorted_inds = _sort_inds(graph, inner_labels, outer_labels, False)
    inner_labels = inner_labels[sorted_inds]
    outer_labels = outer_labels[sorted_inds]

    inner_freq, inner_freq_cumsum, outer_freq, outer_freq_cumsum = _get_freqs(
        inner_labels, outer_labels
    )
    inner_unique, _ = _unique_like(inner_labels)
    outer_unique, _ = _unique_like(outer_labels)

    n_verts = graph.shape[0]
    axline_kws = dict(linestyle="dashed", lw=0.9, alpha=0.3, zorder=3, color="grey")
    # draw lines
    for x in inner_freq_cumsum[1:-1]:
        ax.vlines(x, 0, n_verts + 1, **axline_kws)
        ax.hlines(x, 0, n_verts + 1, **axline_kws)

    # add specific lines for the borders of the plot
    pad = 0.001
    low = pad
    high = 1 - pad
    ax.plot((low, low), (low, high), transform=ax.transAxes, **axline_kws)
    ax.plot((low, high), (low, low), transform=ax.transAxes, **axline_kws)
    ax.plot((high, high), (low, high), transform=ax.transAxes, **axline_kws)
    ax.plot((low, high), (high, high), transform=ax.transAxes, **axline_kws)

    # generic curve that we will use for everything
    lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 500)
    tan = np.tan(lx)
    curve = np.hstack((tan[::-1], tan))

    divider = make_axes_locatable(ax)

    # inner curve generation
    inner_tick_loc = inner_freq.cumsum() - inner_freq / 2
    inner_tick_width = inner_freq / 2
    # outer curve generation
    outer_tick_loc = outer_freq.cumsum() - outer_freq / 2
    outer_tick_width = outer_freq / 2

    # top inner curves
    ax_x = divider.new_vertical(size="5%", pad=0.0, pack_start=False)
    ax.figure.add_axes(ax_x)
    _plot_brackets(
        ax_x,
        np.tile(inner_unique, len(outer_unique)),
        inner_tick_loc,
        inner_tick_width,
        curve,
        "inner",
        "x",
        n_verts,
        fontsize,
    )
    # side inner curves
    ax_y = divider.new_horizontal(size="5%", pad=0.0, pack_start=True)
    ax.figure.add_axes(ax_y)
    _plot_brackets(
        ax_y,
        np.tile(inner_unique, len(outer_unique)),
        inner_tick_loc,
        inner_tick_width,
        curve,
        "inner",
        "y",
        n_verts,
        fontsize,
    )

    if plot_outer:
        # top outer curves
        pad_scalar = 0.35 / 30 * fontsize
        ax_x2 = divider.new_vertical(size="5%", pad=pad_scalar, pack_start=False)
        ax.figure.add_axes(ax_x2)
        _plot_brackets(
            ax_x2,
            outer_unique,
            outer_tick_loc,
            outer_tick_width,
            curve,
            "outer",
            "x",
            n_verts,
            fontsize,
        )
        # side outer curves
        ax_y2 = divider.new_horizontal(size="5%", pad=pad_scalar, pack_start=True)
        ax.figure.add_axes(ax_y2)
        _plot_brackets(
            ax_y2,
            outer_unique,
            outer_tick_loc,
            outer_tick_width,
            curve,
            "outer",
            "y",
            n_verts,
            fontsize,
        )
    return ax


def _plot_brackets(
    ax, group_names, tick_loc, tick_width, curve, level, axis, max_size, fontsize
):
    for x0, width in zip(tick_loc, tick_width):
        x = np.linspace(x0 - width, x0 + width, 1000)
        if axis == "x":
            ax.plot(x, -curve, c="k")
            ax.patch.set_alpha(0)
        elif axis == "y":
            ax.plot(curve, x, c="k")
            ax.patch.set_alpha(0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(axis=axis, which="both", length=0, pad=7)
    for direction in ["left", "right", "bottom", "top"]:
        ax.spines[direction].set_visible(False)
    if axis == "x":
        ax.set_xticks(tick_loc)
        ax.set_xticklabels(group_names, fontsize=fontsize, verticalalignment="center")
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.xaxis.labelpad = 30
        ax.set_xlim(0, max_size)
        ax.tick_params(axis="x", which="major", pad=5 + fontsize / 4)
    elif axis == "y":
        ax.set_yticks(tick_loc)
        ax.set_yticklabels(group_names, fontsize=fontsize, verticalalignment="center")
        ax.set_ylim(0, max_size)
        ax.invert_yaxis()
