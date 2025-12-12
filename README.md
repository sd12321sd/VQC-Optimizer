## Overview

The Variational Quantum Eigensolver (VQE) is one of the most prominent algorithms for finding ground states of Hamiltonians on near-term noisy intermediate-scale quantum (NISQ) devices. In VQE, a classical optimizer is used to update the parameters of a variational quantum circuit, and its performance plays a crucial role in determining the accuracy and efficiency of the algorithm. Consequently, designing optimizers that are tailored to variational quantum circuits is essential for improving VQE performance and for unlocking its potential quantum advantage.

This repository contains the core implementations of optimization methods proposed in the manuscripts listed below. The repository is currently under active development. In future updates, we plan to provide **PennyLane-based implementations** of these methods, enabling them to be easily used as APIs in practical VQE workflows.

## Related Papers

The methods implemented (or planned to be implemented) in this repository are based on the following works:

1. **WA-QNG** — *Weighted Approximate Quantum Natural Gradient for Variational Quantum Eigensolver*  
   arXiv:2504.04932

2. **H-QNG** — *Efficient Hamiltonian-aware Quantum Natural Gradient Descent for Variational Quantum Eigensolvers*  
   arXiv:2511.14511

## Status

🚧 This repository is under construction.  
Core algorithmic implementations are being added, and a PennyLane-based API will be provided in future releases.
