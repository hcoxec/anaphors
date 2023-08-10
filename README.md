# Anaphoric Structure Emerges Between Neural Networks

This repository contains code for the CogSci 2023 paper [Anaphoric Structure Emerges Between Neural Networks](https://escholarship.org/content/qt6qf6251k/qt6qf6251k.pdf) by Nicholas Edwards, Hannah Rohde &amp; Henry Conklin.

### Setup
Create a new virtual environment and run:

```bash scripts/setup.sh```

to install required packages. Alternatively, install the packages in **scripts/requirements.txt** however you like.

### Running Jobs
Each run needs a config file, the config used for the experiments in the paper can be found under **configs/cogsci_2023.jsonnet**

Each run relies on **code/train.py**, where all the data preprocessing (either loading/generation) occurs, followed by the actual model training.

Included under the scripts directory is a file **run.sh** which runs this by calling:

```bash run.sh```

Also included is **run_array.sh** which runs the model for 10 initialisations and provides an example of how to add commandline arguments for running dependent parameters, like random seed.


### General

1.  Data is provided in the ```data/redundant_predicates``` directory.
- 'redlarge_{train/val/test}': multi-agent experiment data
- 'redlarge_{comp/tok/null}_comp_{train/val/test}': learnability data (in each handcrafted language: 'No Elision'/'Pronoun'/'Pro-drop' respectively)

2. Code for the main multi-agent experiments (section "Languages with anaphoric structure emerge between neural agents") is found in the ```code``` directory.

Hyperparameters used in our experiments:

1. Language
- `signal_alphabet_size`: size of the vocabulary in the communication channel (26)
- `signal_len`: maximum length of a message generated by Sender in the communication channel (10)
- `n_roles`: number of roles in the meaning (5)
- `n_atoms`: number of total atoms (or 'words') in the meaning space (32)
- `length_cost`: penalty applied to message length in the loss function, expressed as $\alpha$ (0.0 for Control condition; 0.15 for Efficiency condition)

2. Agents
- `rnn_cell`: the type of Sender and Receiver recurrent cell (`gru`)
- `hidden_size`: size of the hidden layer for Sender and Receiver (250)
- `embedding_size`: dimensionality of the embedding hidden layer for Sender and Receiver (32, i.e., equal to the meaning space size)

3. Optimisation
- `max_epochs`: number of training epochs (3000)
- `batch_size`: size of a training batch (5000)
- `learning_rate`: learning rate (0.001)
- `optimizer`:  type of optimization method used (`adam`)
- `data_scaling`: factor by which to scale up data, making it appear larger than it is - intended for use with REINFORCE, where large batch sizes mean epochs are too fast (20)
- `sender_entropy`: entropy regularisation coefficient for Sender (0.5)

For evaluation, use the notebook ```interaction_checkpoint_eval.ipynb```.
```interaction_config.jsonnet``` specifies the config used in the multi-agent experiments whose checkpoints are being evaluated. This file also imports functions from ```analyser.py```, which provides support for computing the measures outlined in the paper.


3.  Code for the first section of the paper "Neural agents can learn languages with anaphoric structure" can be found as Jupter notebooks in the ```experiments``` directory:
 - ```learnability_expts.ipynb``` (training Receiver, computing Predictive Ambiguity)
 - ```learnability_eval.ipynb``` (plotting loss curves)
 - ```su_handcrafted_langs.ipynb``` (computing signal uniqueness for handcrafted languages)

*N.B.: learnability code still to be finalised, so there may be bugs with loading data, etc.*

#### Attribution

This code is made available under the MIT software license, and uses portions of the code from the Facebook Research [EGG](https://github.com/facebookresearch/EGG) Repo made available under the same license. Notes are made in the code where objects are based on EGG.
