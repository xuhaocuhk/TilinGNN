# TilinGNN
TilinGNN: Learning to Tile with Self-Supervised Graph Neural Network (SIGGRAPH 2020)
[Project page](https://appsrv.cse.cuhk.edu.hk/~haoxu/projects/TilinGnn/index.html)

## We are still refactoring the code, please wait for a few days :)

### Dependencies:
- Pytorch:
https://pytorch.org/get-started/locally/
- Shapely:
pip install Shapely
- torch Geometric
$ pip install --verbose --no-cache-dir torch-scatter
$ pip install --verbose --no-cache-dir torch-sparse
$ pip install --verbose --no-cache-dir torch-cluster
$ pip install --verbose --no-cache-dir torch-spline-conv (optional)
$ pip install torch-geometric
- PyQT5:
pip install PyQt5
- Minizinc (for integer programming solvers)


# How to use
We have the following entries to experiment our project:
- UI interface.
- using our pre-trained models (or IP solver) to tile a given region using a given tile set.
- train a new model (create new tileset, including new superset, generate training data, then training)
