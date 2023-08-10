#! /bin/bash

set -e

cd ..

for i in $(seq 1 10);
do
    echo "Running Seed ${i}"
    python code/train.py configs/cogsci_2023copy.jsonnet --config_args "{\"att\":${i}}"

done