### We are refactoring our code and adding documentation, to improve the clarification. Please wait for a few more days :)

# [TilinGNN: Learning to Tile with Self-Supervised Graph Neural Network (SIGGRAPH 2020)](https://appsrv.cse.cuhk.edu.hk/~haoxu/projects/TilinGnn/index.html)
![Teaser Figure](./images/teaser.png)

### About
The goal of our research problem is illustrated below: given a tile set (a) and a 2D region to be filled (b), we aim to produce a tiling (c) that maximally covers the interior of the given region without overlap or hole between the tile instances.
![](./images/problem.png)

### Dependencies:
- [Pytorch](https://pytorch.org/get-started/locally/)
- [Shapely](https://pypi.org/project/Shapely/)
- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [PyQT5](https://pypi.org/project/PyQt5/)
- [Minizinc](https://pypi.org/project/minizinc/) [Optional, install it only when you use IP solvers]

### Usage
We provide the following entry points for researchers to try our project:
- **Tiling Design by UI interface**: From file `Tiling-GUI.py`, you can use our interface to draw a tiling region, and preview the tiling results interactively.  
- **Tiling a region of silhouette image**: From file `Tiling-Shape.py`, you can use our pre-trained models, or IP solver, to solve a tiling problem by specifying a tiling region (from silhouette image) and a tile set.
- **Training for new tile Sets**: You need the following steps to train a network for a new tile set. 
    1. Following the file organization of existing tile sets inside the `data` folder, create a new folder with new files that describe your new tile sets. After that, you need to edit the global configuration file `inputs/config.py` to let the system know you your new tile set.
    1. Create a superset of candidate tile placements by running file `tiling/gen_complete_super_graph.py`, the generated files will be stored in the folder you created in Step (1).
    1. Generate training data of random shapes by running `solver/ml_solver/gen_data.py`, the data will be stored in the path recorded in file `inputs/config.py`.
    1. Start network training by running file `solver/ml_solver/network_train.py`.
    
#### Note
In this program, we have a global configuration file `inputs/config.py`, which plays very important role to control the behavior of the programs, such as which tile set you want to work with, the stored location of the trained networks, or how many training data you will create, etc.   
