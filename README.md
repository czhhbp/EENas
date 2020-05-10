# EENas:Efficient Evolution for Neural Architecture Search

# Introduction
We proposed an efficient evolution method of neural architecture search. Our method adopts the weight sharing strategy, in which a supernet is built to subsume all architectures, to speed up architecture evaluation. A universal choice strategy is designed to deal with the inaccurate evaluation caused by the methods that speeding up evaluation. Instead of searching for the architecture with the best inaccurate performance, we evolve and improve the population of excellent architectures and derive the final architecture according to commonalities of these architectures. The proposed method achieved the better results(2.40\% test error rate on CIFAR-10 with 3.66M parameters) compared to other the-state-of-art method using less than 0.4 GPU days.

# Requirements
Python 3.6 +

Pytorch 1.0.1 +

# Search and Evaluate
To search,
'''
python search_train.py
'''
To evaluate,
Change 'genome' of darts-codes/genotypes.py into the architecture found, then run:
'''
cd darts_codes && python train.py
'''

We use partial [Darts](https://github.com/quark0/darts) codes as our evaluation environment so as to ensure fair comparison.

# Results
WE found final architecture as shown below and provide the trained network weights 'best_weights.pt' on CIFAR-10.
![cells](https://github.com/czhhbp/EENas/blob/master/cells.png)
