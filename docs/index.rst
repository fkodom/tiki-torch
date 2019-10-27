
Tiki
====

Train your neural networks on island time. 🏄🌴🌞🍹.


Contents
--------

.. toctree::
   :maxdepth: 2

   tutorials.rst
   api.rst


About
-----

Tiki is a high-level machine learning library written in Python and running on top of PyTorch.  It simplifies the process of training neural network models, allowing you to just focus on the model design.  It provides a concise interface to PyTorch with convenient features:

    * Automatic network training
    * Multi-GPU and multi-node training
    * Built-in visualization dashboard
    * Custom callback functions
    * Tensorboard integration
    * Pre-compiled loss functions

Tiki integrates seamlessly with existing PyTorch models.  It will train *any* ``torch`` object that has both:

    * A defined ``forward`` method
    * One or more trainable parameters

Tiki nearly eliminates the need for boilerplate code, and enables faster experimentation.  By building on top of PyTorch, we provide several user features that other high-level libraries (e.g. Keras) cannot.

    * Avoid using "backend" math libraries
    * Dynamically execute model code
    * Execute and train models `faster than Keras <https://wrosinski.github.io/deep-learning-frameworks/>`_
    * Print or set breakpoints from within a model
    * Simplified code debugging


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
