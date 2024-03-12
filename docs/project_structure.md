
# Project Structure


## `docs` directory

This directory contains different files used to provide additional information beyond the README. An `imgs` subdirectory contains images used in the README and docs.


## `notebooks` directory

This directory contains Jupyter notebooks used for the project. Code to generate many of the figures seen in the paper can be found in a few of these notebooks.


## `scripts` directory

This directory contains various scripts for the project including the following:

- `eval_pybullet_data.py`: evaluates a method on a pybullet dataset
- `train_neural_network.py`: trains a PointSDF-like neural network


## `src/v_prism/data_loading` directory

This directory contains code for loading in data and negative sampling. There are also functions provided for downloading the different datasets used in the project.


## `src/v_prism/mapping` directory

This directory contains code for creating maps from data. There is code for the V-PRISM algorithm, the BHM algorithm, and variants with gradient descent.


## `src/v_prism/utils` directory

This directory contains utility functions and classes for the rest of the project. Its contents range from kernel definitions, UI utilities, and RANSAC implementations.


## `tests` directory

This directory contains some tests written to ensure that the code is not egregiously wrong. It requires the pytest module to run, which is not part of the conda environment. Thus, you must run `pip install pytest` before running `pytest ./tests` if you have not already.


