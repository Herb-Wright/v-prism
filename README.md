
# V-PRISM: Probabilistic Mapping of Unknown Tabletop Scenes

![](./docs/imgs/fig1.png)


## Overview

This is the code for the paper "V-PRISM: Probabilistic Mapping of Unknown Tabletop Scenes". Our method builds a 3D probabilistic map of a tabletop scene by using a feature transform and performing an EM algorithm. We employ a novel negative sampling technique before our feature transform to fully encode information about the scene. Our method not only reconstructs the scene accurately, but encodes valuable information about uncertainty due to occlusion. In our paper, we perform both qualitative and quantitative experiments to verify these claims. See below for a figure outlining our method.

![](./docs/imgs/fig2.png)


## Usage

**Getting Started:**

1. clone the repo
    ```bash
    git clone https://github.com/Herb-Wright/v-prism.git v_prism
    cd v_prism
    ```
2. create the conda environment (if you don't have cuda, use `env_cpu.yml` and `v_prism_cpu` instead):
    ```bash
    conda env create -f environment.yml
    conda activate v_prism
    ```
3. install the project
    ```bash
    pip install -e .
    ```

**Using VPRISM:**

Here is an example:

```python
from v_prism import full_VPRISM_method

map = full_VPRISM_method(X, y, num_classes, cam_pos)
```

See `./notebooks/example_demo.ipynb` for a more detailed example.

**Running Experiments:**

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

## Resources

Please consider citing our work:

```
# TODO
```

See the `./docs` directory for more information








