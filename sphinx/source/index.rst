.. cslrtools2 documentation master file, created by
   sphinx-quickstart on Thu Nov 13 14:59:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cslrtools2 documentation
========================

**Comprehensive toolkit for Continuous Sign Language Recognition (CSLR) research**

cslrtools2 provides landmark extraction pipelines, dataset management utilities, and PyTorch helpers for sign language video analysis.

Features
--------

* **LMPipe**: Landmark extraction pipeline using MediaPipe
* **SLDataset**: Sign language dataset management with Zarr
* **ConvSize**: PyTorch convolution size calculation utilities

Installation
------------

.. code-block:: bash

   pip install cslrtools2

Quick Start
-----------

Extract landmarks from a video:

.. code-block:: python

   from cslrtools2.lmpipe import LMPipeInterface
   from cslrtools2.plugins.mediapipe.lmpipe import MediaPipeHolistic

   # Initialize estimator
   estimator = MediaPipeHolistic()

   # Create pipeline
   interface = LMPipeInterface(estimator)

   # Process video
   interface.run_single_video("video.mp4", "output.npz")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules
   cslrtools2

.. toctree::
   :maxdepth: 1
   :caption: Links:

   GitHub Repository <https://github.com/ikegami-yukino/python-cslrtools2>
   PyPI Package <https://pypi.org/project/cslrtools2/>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


