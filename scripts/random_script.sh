#!/bin/bash

# get the scripts directory
SCRIPTS_DIR="$(dirname $0)"

python scripts/eval_pybullet_data.py -d bhm4recon_shapenet_100 -a bad_sampling -o bad_sampling_shapenet_100.txt
python scripts/eval_pybullet_data.py -d bhm4recon_objaverse_100 -a bad_sampling -o bad_sampling_objaverse_100.txt


python scripts/eval_pybullet_data.py  -d bhm4recon_shapenet_100 -a no_stratified -o stratified_shapenet_100.txt
python scripts/eval_pybullet_data.py  -d bhm4recon_ycb_100 -a no_stratified -o stratified_shapenet_100.txt
python scripts/eval_pybullet_data.py  -d bhm4recon_objaverse_100 -a no_stratified -o stratified_shapenet_100.txt

python scripts/eval_pybullet_data.py  -d bhm4recon_shapenet_100 -a no_under_table -o under_shapenet_100.txt
python scripts/eval_pybullet_data.py  -d bhm4recon_ycb_100 -a no_under_table -o under_shapenet_100.txt
python scripts/eval_pybullet_data.py  -d bhm4recon_objaverse_100 -a no_under_table -o under_shapenet_100.txt

