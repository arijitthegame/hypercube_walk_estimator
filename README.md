# Walking the Hypercube: Unbiased Quantum Partition Functions Without the Matrix

## An unbiased estimator for the partition function of a quantum spin Hamiltonian which runs in linear time.

In this work, we propose an unbiased stochastic estimator of the partition function using only local random walks, avoiding both matrix materialization and matrix-vector products. Our approach combines Graph Random Features with the Hutchinson’s stochastic trace estimator, and exploits the algebraic structure of quantum spin Hamiltonians to reduce the per-step computational cost to O(n + d), where d is the mean degree of the spin coupling graph, independent of the Hilbert space dimension N = 2n. 

The repo accompanies the paper ["Walking the Hypercube: Unbiased Quantum Partition Functions Without the Matrix"](https://openreview.net/forum?id=wLHljLWRDA)

Avinava Dubey*, Arijit Sehanobish*, Krzysztof Choromanski*

Google Research, Kensho Technologies, Google DeepMind, Columbia University,

 40th Conference on Neural Information Processing Systems (NeurIPS 2026).



## Installation
```bash
git clone https://github.com/arijitthegame/hypercube_walk_estimator.git
cd super_graph_random_features
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
pip3 install -e . --user
```

## Getting Started
All code resides in the `src` folder.  


## Citation
If you find our work useful, please cite : 

```bibtex
@inproceedings{
anonymous2026walking,
title={Walking the Hypercube: Unbiased Quantum Partition Functions Without the Matrix},
author={Kumar Avinava Dubey and Arijit Sehanobish and Krzysztof Marcin Choromanski},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=wLHljLWRDA}
}
# GRF_Quantum_Physics
GRF For Quantum Physics
