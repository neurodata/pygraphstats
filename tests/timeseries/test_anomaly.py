import numpy as np
import pytest
from numpy.testing import assert_equal

from graspologic.timeseries import anomaly_detection


def load_data():
    f = np.load("tests/timeseries/data/anomaly.npz")
    graphs = f["graphs"]
    return graphs


def test_inputs():
    graphs = load_data()

    with pytest.raises(TypeError):
        anomaly = anomaly_detection(graphs, method=1)

    with pytest.raises(ValueError):
        anomaly = anomaly_detection(graphs, method="LaplacianSpectralEmbed")

    with pytest.raises(TypeError):
        anomaly = anomaly_detection(graphs, time_window="3")

    with pytest.raises(ValueError):
        anomaly = anomaly_detection(graphs, time_window=20)  # There are 12 input graphs

    with pytest.raises(ValueError):
        anomaly = anomaly_detection(graphs, time_window=1)

    with pytest.raises(TypeError):
        anomaly = anomaly_detection(graphs, use_lower_line="True")


def test_pipeline():
    """Based on the example in Guodong's repo"""

    def run_anomaly(method, use_lower_line):
        graphs = load_data()

        (graph_indices, graph_dict, vertex_indices, vertex_dict) = anomaly_detection(
            graphs,
            method=method,
            time_window=4,
            n_components=1,
            use_lower_line=use_lower_line,
        )

        return graph_indices, vertex_indices

    # Use Omni, no lower line.
    g_idx, v_idx = run_anomaly("omni", use_lower_line=False)

    # Expected graphs to be different
    assert_equal(g_idx, [5, 10])
    # All 100 vertices in graphs 5 and 6 should be different
    assert_equal(len(v_idx[5]), 100)
    assert_equal(len(v_idx[6]), 100)

    # Use Omni, use lower line
    g_idx, v_idx = run_anomaly("omni", use_lower_line=True)
    # Expected graphs to be different
    assert_equal(g_idx, [4, 5, 10])
    # All 100 vertices in graphs 5 and 6 should be different
    assert_equal(len(v_idx[5]), 100)
    assert_equal(len(v_idx[6]), 100)

    # Use MASE, no lower_line
    g_idx, v_idx = run_anomaly("mase", use_lower_line=False)

    # Expected graphs to be different
    assert_equal(g_idx, [5])
    # All 100 vertices in graphs 5 and 6 should be different
    assert_equal(len(v_idx[5]), 100)
    assert_equal(len(v_idx[6]), 100)

    # Use MASE, no lower_line
    g_idx, v_idx = run_anomaly("mase", use_lower_line=True)

    # Results are the same
    # Expected graphs to be different
    assert_equal(g_idx, [5])
    # All 100 vertices in graphs 5 and 6 should be different
    assert_equal(len(v_idx[5]), 100)
    assert_equal(len(v_idx[6]), 100)
