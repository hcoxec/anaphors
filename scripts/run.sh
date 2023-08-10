#! /bin/bash

set -e

cd ..

python code/train.py configs/cogsci_2023copy.jsonnet
