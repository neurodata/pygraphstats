********
Tutorial
********

.. _models_tutorials:

Models
======
This tutorial presents several random graph models: the Erdos-Renyi (ER) model, degree-corrected ER model,
stochastic block model (SBM), degree-corrected SBM, and random dot product graph model. These models provide a basis for studying random graphs. All models are shown fit to the same dataset.

.. toctree::
   :maxdepth: 1
      
   tutorials/models/models

.. _simulations_tutorials:

Simulations
===========
The following tutorials demonstrate how to easily sample random graphs from graph models such as the Erdos-Renyi model, 
stochastic block model, and random dot product graph (RDPG).

.. toctree::
   :maxdepth: 1
   
   tutorials/simulations/erdos_renyi
   tutorials/simulations/sbm
   tutorials/simulations/rdpg
   tutorials/simulations/corr
   tutorials/simulations/rdpg_corr

.. _embed_tutorials:

Embedding
=========
Inference on random graphs depends on low-dimensional Euclidean representation of the vertices of graphs, known as *graph embeddings*, typically given by spectral decompositions of adjacency or Laplacian matrices. Below are tutorials for computing graph embeddings of single graph and multiple graphs.

.. toctree::
   :maxdepth: 1
   
   tutorials/embedding/AdjacencySpectralEmbed
   tutorials/embedding/Omnibus
   
.. _inference_tutorials: 

Inference
===========================
Statistical testing on graphs requires specialized methodology in order to account 
for the fact that the edges and nodes of a graph are dependent on one another. Below 
are tutorials for robust statistical hypothesis testing on multiple graphs.

.. toctree::
   :maxdepth: 1

   tutorials/inference/latent_position_test
   tutorials/inference/latent_distribution_test

.. _plot_tutorials:

Plotting
========
The following tutorials present ways to visualize the graphs, such as its adjacency matrix, and graph embeddings. 

.. toctree::
   :maxdepth: 1

   tutorials/plotting/heatmaps
   tutorials/plotting/gridplot
   tutorials/plotting/pairplot

.. _matching_tutorials:

Matching
========
The following tutorials demonstrate how to use the graph matching functionality,
including an introduction to the module, and how to utilize the seeding feature.

.. toctree::
   :maxdepth: 1

   tutorials/matching/faq
   tutorials/matching/sgm
   tutorials/matching/padded_gm

.. _subgraph_tutorials:

Subgraph
========
Statistical estimation of the signal subgraph for a set of graphs of multiple classes,
done iteratively, non-iteratively, and with coherence. Below are tutorials for finding
the estimated signal subgraph.

.. toctree::
   :maxdepth: 1
      
   tutorials/subgraph/Screen
   tutorials/subgraph/ItScreen
   tutorials/subgraph/Coherence
