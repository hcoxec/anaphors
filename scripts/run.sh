#! /bin/bash

set -e

cd ../code

python main.py preprocess ../configs/cogsci_2023.jsonnet

python main.py train ../configs/cogsci_2023.jsonnet

python main.py eval ../configs/cogsci_2023.jsonnet