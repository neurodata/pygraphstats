import numpy as np
import random
from graspologic.simulations import sbm
from graspologic.subgraph import Screen

np.random.seed(10)


def data_generator(num_graphs, N, n, prob_tensor, percent_vec):

    # Getting the number of classes
    num_types = len(percent_vec)

    # Getting vector with the number of graphs in each class
    num = [int(num_graphs * a) for a in percent_vec]

    # Creating blank arrays for all of the returns
    data = np.zeros((num_graphs, N, N))
    y_label = np.zeros((num_graphs, 1))

    # Creates vector of random indices to randomly distribute graphs in tensor
    L_ind = random.sample(range(0, num_graphs), num_graphs)

    # Loop for creating the returns
    for i in range(num_types):

        # Create tensor that will contain all of the graphs of one type
        types = np.zeros((num[i], N, N))

        # Put all the graphs of one type into types
        for j in range(len(types)):
            types[j] = sbm(n=n, p=prob_tensor[i])

        # Assigns all of the graphs in types to random indices in data
        data[L_ind[: num[i]]] = types

        # Creates corresponding labels
        y_label[L_ind[: num[i]]] = int(i)

        # Gets rid of used indices
        L_ind = L_ind[num[i] :]

    return data, y_label


# Generate data
prob_tensor = np.zeros((2, 2, 2))
prob_tensor[0] = [[0.3, 0.2], [0.2, 0.3]]
prob_tensor[1] = [[0.4, 0.2], [0.2, 0.3]]
n = [20, 180]
percent_vec = np.asarray([0.50, 0.50])
data_test, y_label_test = data_generator(100, 200, n, prob_tensor, percent_vec)


def test_non_itss_mgc():

    # Testing mgc
    screen = Screen("mgc", float("-inf"))
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing if S_hat_samp has correct values and shape
    assert np.array_equal(S_hat_samp, data_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)


def test_non_itss_dcorr():

    # Redo for dcorr
    screen = Screen("dcorr", float("-inf"))
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing if S_hat_samp has correct values and shape
    assert np.array_equal(S_hat_samp, data_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)


def test_non_itss_rv():

    # Redo for rv
    screen = Screen("rv", float("-inf"))
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing if S_hat_samp has correct values and shape
    assert np.array_equal(S_hat_samp, data_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)


def test_non_itss_cca():

    # Redo for cca
    screen = Screen("cca", float("-inf"))
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing if S_hat_samp has correct values and shape
    assert np.array_equal(S_hat_samp, data_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)
