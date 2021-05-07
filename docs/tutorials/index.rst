*********
Tutorials
*********
.. toctree::
   :hidden:

   Reference Documentation <https://graspologic.readthedocs.io/en/latest/>

.. _models_tutorials:

Models
======
This tutorial presents several random graph models: the Erdos-Renyi (ER) model, degree-corrected ER model,
stochastic block model (SBM), degree-corrected SBM, and random dot product graph model. These models provide a basis for studying random graphs. All models are shown fit to the same dataset.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   models/models

.. _simulations_tutorials:

Simulations
===========
The following tutorials demonstrate how to easily sample random graphs from graph models such as the Erdos-Renyi model,
stochastic block model, and random dot product graph (RDPG).

.. toctree::
   :maxdepth: 1
   :titlesonly:

   simulations/erdos_renyi
   simulations/sbm
   simulations/mmsbm
   simulations/rdpg
   simulations/corr
   simulations/rdpg_corr

.. _cluster_tutorials:

Clustering
==========
The following tutorials explain how to cluster vertex or graph embeddings with two
clustering algorithms, as well as the advantages of these to comparable implementations.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   clustering/autogmm
   clustering/kclust

.. _embed_tutorials:

Embedding
=========
Inference on random graphs depends on low-dimensional Euclidean representation of the vertices of graphs, known as *graph embeddings*, typically given by spectral decompositions of adjacency or Laplacian matrices. Below are tutorials for computing graph embeddings of single graph and multiple graphs.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   embedding/AdjacencySpectralEmbed
   embedding/OutOfSampleEmbed
   embedding/CovariateAssistedEmbed
   embedding/MASE
   embedding/Omnibus

.. _inference_tutorials:

Inference
===========================
Statistical testing on graphs requires specialized methodology in order to account
for the fact that the edges and nodes of a graph are dependent on one another. Below
are tutorials for robust statistical hypothesis testing on multiple graphs.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   inference/latent_position_test
   inference/latent_distribution_test

.. _plot_tutorials:

Plotting
========
The following tutorials present ways to visualize the graphs, such as its adjacency matrix, and graph embeddings.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   plotting/heatmaps
   plotting/gridplot
   plotting/pairplot
   plotting/matrixplot
   plotting/pairplot_with_gmm

.. _matching_tutorials:

Matching
========
The following tutorials demonstrate how to use the graph matching functionality,
including an introduction to the module, and how to utilize the seeding feature.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   matching/faq
   matching/sgm
   matching/padded_gm

.. _subgraph_tutorials:

Subgraph
========
The following tutorial demonstrates how to estimate the signal-subgraph of samples of a graph/class model according to either the coherent or incoherent estimator models.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   subgraph/subgraph

.. _vertex_nomination_tutorials:

Vertex Nomination
=================
The following tutorials demonstrate how to use unattributed single graph spectral vertex nomination or vertex nomination via seeded graph matching to find vertices that are related to a given vertex / set of vertices of interest.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   vertex_nomination/SpectralVertexNomination
   nominate/vertex_nomination_via_SGM

.. _aligning_tutorials:

Aligning
========
The following tutorials shows how to align two seperate datasets with each other, for better comparison of the data.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   aligning/aligning

.. _connectomics_tutorials:

Connectomics
============
The following tutorials demonstrate how to apply methods in this package to the analysis of connectomics datasets.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   connectomics/mcc
