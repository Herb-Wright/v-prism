---
layout: ../layouts/MarkdownLayout.astro
---


# V-PRISM Code Documentation

## Setup

### Requirements

- Conda/Mamba
- Cuda GPU (optional)

### Steps

1. clone the repo
    ```bash
    git clone https://github.com/Herb-Wright/v-prism.git v_prism
    cd v_prism
    ```
2. create the conda environment (if you don't have cuda, use `env_cpu.yml` instead):
    ```bash
    conda env create -f environment.yml
    ```
3. install the project
    ```bash
    pip install -e .
    ```

## Examples of Using V-PRISM

Here is an example of using the full method:

```python
from v_prism import full_VPRISM_method

map = full_VPRISM_method(
    X,  # (P, 3) Tensor
    y,  # (P,) Tensor
    num_classes,
    cam_pos,  # (3,) Tensor
)
```

Here is an example of negative sampling:

```python
from v_prism import full_negative_sampling_method

X, y = full_negative_sampling_method(
    points,  # (P, 3) Tensor
    seg_mask,  # (P,) Tensor
    ray_step_size,
    object_sphere_radius,
    camera_pos,  # (3,) Tensor
    scene_sphere_radius,
    subsample_grid_size_unocc,
    subsample_grid_size_occ,
):
```



## Running Experiments


1. download one the following datasets:
    - ShapeNet Scents
    - YCB Scenes
    - Objaverse Scenes  
2. run the script

If you want to evaluate PointSDF: download pt file or train


*\*Links coming soon*


## Project Structure


### `docs` directory

This contains the code for the website.

### `notebooks` directory

This directory contains Jupyter notebooks used for the project. Code to generate many of the figures seen in the paper can be found in a few of these notebooks.


### `scripts` directory

This directory contains various scripts for the project including the following:

- `eval_pybullet_data.py`: evaluates a method on a pybullet dataset
- `train_neural_network.py`: trains a PointSDF-like neural network


### `src/v_prism/data_loading` directory

This directory contains code for loading in data and negative sampling. There are also functions provided for downloading the different datasets used in the project.


### `src/v_prism/mapping` directory

This directory contains code for creating maps from data. There is code for the V-PRISM algorithm, the BHM algorithm, and variants with gradient descent.


### `src/v_prism/utils` directory

This directory contains utility functions and classes for the rest of the project. Its contents range from kernel definitions, UI utilities, and RANSAC implementations.


### `tests` directory

This directory contains some tests written to ensure that the code is not egregiously wrong. It requires the pytest module to run, which is not part of the conda environment. Thus, you must run `pip install pytest` before running `pytest ./tests` if you have not already.


## [>> Go To Github](https://github.com/Herb-Wright/v-prism)


