### We are refactoring our code and adding documentation, to improve the clarification. Please wait for a few more days :)

# TilinGNN
TilinGNN: Learning to Tile with Self-Supervised Graph Neural Network (SIGGRAPH 2020)
[Project page](https://appsrv.cse.cuhk.edu.hk/~haoxu/projects/TilinGnn/index.html)

### Dependencies:
- [Pytorch](https://pytorch.org/get-started/locally/)
- [Shapely](https://pypi.org/project/Shapely/)
- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [PyQT5](https://pypi.org/project/PyQt5/)
- [Minizinc](https://pypi.org/project/minizinc/)

# How to use
We have the following entries for you to experience our project:
- **Tiling Design by UI interface**: From file TilinGUI.py, you can use our interface to draw a tiling region, and preview the tiling result interactively.  
- **Tiling a Region of Silhouette Image**: From file XXX, you can use our pre-trained models (or IP solver) to tile a tiling region selected from silhouette image.
- **Training for New Tile Sets**: You need the following steps to train a network for a new tile set. 
    - create your new tile sets by XXX, from file XXX
    - generate candidate tile placements by XXX, the files will be stored as XXX
    - generate training data by XXX 
    - train new model from file XXX   
