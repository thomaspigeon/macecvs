# macecvs

## Overview

`macecvs` is a Python package that leverages machine learning toolds and the MACE framework for the definitions of collective variable. This package integrates an example in which many short trajectories are generated and from which some small dimensionnal collectives variables are defined using a matrix auto-encoder invariant with respect to permutations of the lines of the matrix. Finally some path based collective variables can be defined to in this collective variables space. 

## Features

- **Custom Calculators**: Modified MACE calculators to integrate MACE descriptor into the defintion of collective variables.
- **Matrix auto-encoder invariant with respect to permutation of lines**: Definition of small dimentionnal CVs from matrix of MACE descriptors, for which each lines are the descriptor of the atomic environment. 
- **Path collective variables**: Some simple tools to create some path collective variables relying on the features of the auto-encoder  
---

## Scripts

### 1. `calculations/calculator.py`
#### Purpose
Defines a custom calculator (`ModifiedMACECalculator`) to return more mace propertires and link with Auto-encoder.

#### Features
- Identifcal to the original MACE calculator but also return the batch and output informations the results dicttionary.

### 2. `models/permutation_invariant_ae.py`
#### Purpose
Defines a matrixa autoencoder invariant with respect to permutation of the lines of the matrix. 

#### Features
- Provides a deep learning architecture.

### 3. `training/train_permutation_invariant_ae.py`
#### Purpose
Specialized training script for the permutation-invariant autoencoder.


### 4. `example/*`
#### Purpose
Various scripts that: 
- generate some trajectories from two sadlle points on the PES and write the MACE descriptors trajectories;
- train the line permutation invariant matrix auto-encoder 
- plot the AE features of some labelled trajectories on top of the training dataset to identify the various zones of the feature space


## Dependencies
- `torch`, `numpy`, `ase`, `e3nn`, `mace`.

## Installation

### 1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/macecvs.git
   ```

### 2. Navigate to the directory:
   ```bash
   cd macecvs
   ```

### 3. Install the package and the dependencies: 
   ```bash
   pip install . 
   ```


