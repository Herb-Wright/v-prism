#!/bin/bash

# get the scripts directory
SCRIPTS_DIR="$(dirname $0)"

python scripts/eval_sim_data.py -d vprism_shapenet_100 -a vprism -o vprism_shapenet_100.txt
python scripts/eval_sim_data.py -d vprism_ycb_100 -a vprism -o vprism_ycb_100.txt
python scripts/eval_sim_data.py -d vprism_objaverse_100 -a vprism -o vprism_objaverse_100.txt

python scripts/eval_sim_data.py -d vprism_shapenet_100 -a bad_sampling -o bad_sampling_shapenet_100.txt
python scripts/eval_sim_data.py -d vprism_ycb_100 -a bad_sampling -o bad_sampling_ycb_100.txt
python scripts/eval_sim_data.py -d vprism_objaverse_100 -a bad_sampling -o bad_sampling_objaverse_100.txt

python scripts/eval_sim_data.py -d vprism_shapenet_100 -a no_stratified -o stratified_shapenet_100.txt
python scripts/eval_sim_data.py -d vprism_ycb_100 -a no_stratified -o stratified_ycb_100.txt
python scripts/eval_sim_data.py -d vprism_objaverse_100 -a no_stratified -o stratified_objaverse_100.txt

python scripts/eval_sim_data.py -d vprism_shapenet_100 -a no_under_table -o under_shapenet_100.txt
python scripts/eval_sim_data.py -d vprism_ycb_100 -a no_under_table -o under_ycb_100.txt
python scripts/eval_sim_data.py -d vprism_objaverse_100 -a no_under_table -o under_objaverse_100.txt
