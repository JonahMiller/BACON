# BACON

This project reproduces Langley’s BACON system for Computational Scientific Discovery. It uses his [book](https://direct.mit.edu/books/book/4759/Scientific-DiscoveryComputational-Explorations-of) as the main inspiration to reimagine his methods using Python. 

## Modern day BACON

This project recreates BACON.1, BACON.3, BACON.5 and BACON.6. It also has functionality for novel methodologies, including my own creations and a Monte-Carlo Tree Search to traverse the space.

The datasets tested on were synthetic and can be found in [datasets.py](./datasets.py). 

In [main.py](./main.py) the full list of methods used in the Space of Data and Space of Laws are shown and can be customised by the user in an args file. For example, to run on the Ideal Gas Law with 3% noise in the dependent variable, the command is:

`python3 main.py --dataset ideal --noise 0.03 --args args/ideal/ideal30.json`

With the args file user customisable.

## Additional comments

This project was completed as a dissertation for the  Part III Computer Science course at the University of Cambridge. The final report is in the [dissertation.pdf](./dissertation.pdf).


## Citation


The full paper is available here:

https://www.researchgate.net/publication/389562521_The_BACON_system_for_equation_discovery_from_scientific_data_Reconciling_classical_artificial_intelligence_with_modern_machine_learning_approaches

The BACON system for equation discovery from scientific data: Reconciling classical artificial intelligence with modern machine learning approaches, Jonah Miller, Soumya Banerjee 4th Annual AAAI Workshop on AI to Accelerate Science and Engineering (AI2ASE), 2025

## Contact

Soumya Banerjee

sb2333@cam.ac.uk
