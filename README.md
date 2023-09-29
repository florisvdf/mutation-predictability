# mutation-predictability

This repository contains the data and code used to generate results and corresponding 
figures of the recently released preprint 
["What makes the effect of protein mutations difficult to predict?"](https://doi.org/10.1101/2023.09.25.559319).

### Project structure
All experiments and processing of results are organized in notebooks, which can be run by 
installing the `predictability` package.

### Installation
Clone the repository and install with
```
git clone https://github.com/florisvdf/mutation-predictability.git
cd mutation-predictability
pip install .
```
The Potts Regressor model of the `predictability` package makes 
use of [`gremlin_cpp`](https://github.com/sokrypton/GREMLIN_CPP). 
To use the Potts Regressor, make sure that `gremlin_cpp` is installed 
and is added to `$PATH`.
