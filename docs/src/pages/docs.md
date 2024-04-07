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

Also checkout the example notebook: [./notebooks/example_demo.ipynb](https://github.com/Herb-Wright/v-prism/blob/main/notebooks/example_demo.ipynb)

## Running Experiments


1. download the items you need to run the particular experiment:
    - ShapeNet Scenes - [download](https://drive.google.com/file/d/1-4KgO3pz7h-sMy7VgjZID6RuzMpCywBm/view?usp=drive_link)
    - YCB Scenes - [download](https://drive.google.com/file/d/1v35PNb-PFOiRGDm1eDwMqRz5uE_fx8kp/view?usp=drive_link)
    - Objaverse Scenes - [download](https://drive.google.com/file/d/1Z2-NPplEDUOLsRgtymlqnHs_HGnMucjt/view?usp=drive_link)
    - ShapeNetCore.v2 Dataset - [website](https://shapenet.org/)
    - YCB Dataset - [website](https://www.ycbbenchmarks.com/)
    - Objaverse .obj files (Not Available)
    - PointSDF weights - [download](https://drive.google.com/file/d/15EILzCZQ1eqfxAGxk0LeGtfBwDyDj-E9/view?usp=drive_link)
2. run the script `python ./scripts/eval_sim_data.py` with desired arguments

**Note.** You might not be able to run objaverse experiments because I had to do some extra steps to convert certain meshes to obj files.


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


