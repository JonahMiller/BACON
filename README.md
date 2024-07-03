# BACON

This project reproduces Langley’s BACON system for Computational Scientific Discovery. It uses his [book](https://direct.mit.edu/books/book/4759/Scientific-DiscoveryComputational-Explorations-of) as the main inspiration to reimagine the algorithms of his methods using modern day Python. 

## Modern day BACON
This project recreates BACON.1, BACON.3, BACON.5 and BACON.6. It also has functionality for novel methodologies, including my own creations and a Monte-Carlo Tree Search to traverse the space.

The datasets tested on were synthetic and can be found in [datasets.py](./datasets.py). In [main.py](./main.py) the full list of methods used in the Space of Data and Space of Laws are shown and can be customised by the user. For example to run on the Ideal Gas Law with 3% noise in the dependent variable, the command is:

`python3 main.py --dataset ideal --noise 0.03 --args args/ideal/ideal30.json`

With the args file user customisable.

## Additional comments

This project was completed as a dissertation for the  Part III Computer Science course at the University of Cambridge. The final report is in the [dissertation.pdf].