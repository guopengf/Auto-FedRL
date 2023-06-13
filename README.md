# Auto-FedRL

Experimental Pytorch Code for the paper ["Auto-FedRL: Federated Hyperparameter Optimization for Multi-institutional Medical Image Segmentation"](https://arxiv.org/abs/2203.06338) (ECCV 2022) 

## Note: Integration with [NVFlare](https://github.com/NVIDIA/NVFlare) is available [here](https://github.com/NVIDIA/NVFlare/tree/dev/research/auto-fed-rl)!

# Requirements

python=3.6\
pytorch=1.4.0

Please refer conda_environment.yml for more dependencies.

# Abstract

Federated learning (FL) is a distributed machine learning technique that enables collaborative model training while avoiding explicit data sharing. The inherent privacy-preserving property of FL algorithms makes them especially attractive to the medical field. However, in case of heterogeneous client data distributions, standard FL methods are unstable and require intensive hyperparameter tuning to achieve optimal performance. Conventional hyperparameter optimization algorithms are impractical in real-world FL applications as they involve numerous training trials, which are often not affordable with limited compute budgets. In this work, we propose an efficient reinforcement learning (RL)-based federated hyperparameter optimization algorithm, termed Auto-FedRL, in which an online RL agent can dynamically adjust hyperparameters of each client based on the current training progress. 


# Run

## Clone this repo
```bash
git clone git@github.com:guopengf/Auto-FedRL.git
```
## Set up conda environment
```bash
cd Auto-FedRL
conda env create -f conda_environment.yml
conda activate flpt14
```

## Train Auto-FedRL

The examples of training command for continuous search (CS) and CS MLP are provided in:

```bash
bash run_exp.sh
```

The commands related to hyperprameter search:
```bash 
--Search 'enable the hyperparameter search, the default is discrete search'
--continuous_search 'enable continous search'
--continuous_search_drl 'enable continous search using MLP'
--rl_nettype 'select the network type from {mlp} for deep RL agent'
--search_lr 'enable client learning rate search'
--search_ne 'enable client iteration search'
--search_aw 'enable aggregation weights search'
--search_slr 'enable server learning rate search'
```

# Ackonwledgements

This code-base uses certain code-blocks and helper functions from [FedMA](https://github.com/IBM/FedMA), 
[Auto-FedAvg](https://arxiv.org/abs/2104.10195) and [Mostafa et al.](https://arxiv.org/abs/1912.13075).

