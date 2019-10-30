
# Tiki-Torch

Train your neural networks on island time.


### About

Tiki-Torch (`tiki`) is a high-level machine learning library written in Python and running on top of PyTorch.  It simplifies the process of training neural network models, allowing you to just focus on the model design.  It provides a concise interface to PyTorch with convenient features:
* Automatic network training
* Multi-GPU and multi-node training
* Built-in visualization dashboard
* Custom callback functions
* Tensorboard integration
* Pre-compiled loss functions

Tiki integrates seamlessly with existing PyTorch models.  It will train *any* `torch` object that has both:
* A defined `forward` method
* One or more trainable parameters

Tiki nearly eliminates the need for boilerplate code, and enables faster experimentation.  By building on top of PyTorch, we provide several user features that other high-level libraries (e.g. Keras) cannot.
* Avoid using "backend" math libraries
* Dynamically execute model code
* Execute and train models [faster than Keras](https://wrosinski.github.io/deep-learning-frameworks/)
* Print or set breakpoints from within a model
* Simplified code debugging


### Installation

Currently, `tiki` is only installable from source.  First, ensure that all of the dependencies in [requirements.txt](https://gitlab.radiancetech.com/radiance-deep-learning/tiki-torch/blob/v0.1/requirements.txt) are installed.  We recommend installing PyTorch using `conda`, according to the official [PyTorch documentation](https://pytorch.org/get-started/locally/#start-locally). 

Then, clone this repository:
```
$ git clone https://gitlab.radiancetech.com/radiance-deep-learning/tiki-torch.git
```
and run `setup.py`:
```
$ cd tiki-torch
$ python setup.py install
```


### Documentation

After installing `tiki`, the Sphinx API documentation is available as a webpage.  It can be accessed by opening a Terminal window, and running the following command:
```
$ tiki docs
```
The Sphinx docs are automatically built each time this command is executed.  This ensures that the documentation is always up-to-date, but unnecessary Sphinx are not added to this repository.


### Training Visualization

**Tiki-Hut** is a built-in dashboard for visualizing training results using `tiki`.  It is similar in to [TensorBoard](https://www.tensorflow.org/tensorboard/) in many ways, but is written in pure Python using the [Streamlit](https://streamlit.io/) library. Tiki also supports TensorBoard integration as a separate service from Tiki-Hut.
  
This greatly improves the code's readability for other Python/ML developers, and it allows users to more easily develop and share their visualization tools with the ML community.

Training metrics are automatically collected by using the `TikiHut` callback. This includes:
* Training / validation losses
* Performance metrics (e.g. accuracy, precision)
* Epoch / batch numbers
* Training hyperparameters
* Model computation graphs
* Parameter histograms

By default, training plots are generated for each loss/metric as a function of training epoch.  Custom plots can also be generated from within the dashboard.
  
The dashboard is displayed as a web app, which can be accessed by:
```
$ tiki hut --logdir=<path-to-log-folder>
```

The `--logdir` argument is optional, and defaults to `./logs` if not provided.


### Examples

Sample training scripts are provided in the [examples](https://gitlab.radiancetech.com/radiance-deep-learning/tiki-torch/tree/v0.1/examples) folder.
