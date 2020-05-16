import torch.nn as nn
from torch.nn import Sequential

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer_dims: list, activation, batch_norm = True):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        layer_dims = [in_dim] + hidden_layer_dims + [out_dim]
        layers = [Linear_trans(layer_dims[i], layer_dims[i+1], activation=activation, batch_norm= batch_norm)
                  for i in range(len(layer_dims)-1)]
        self.mlp = Sequential(*layers)

    def forward(self, x):
        assert x.shape[-1] == self.in_dim
        return self.mlp(x)


class Linear_trans(nn.Module):

    def __init__(self, in_dim, out_dim, activation = None, batch_norm = True):
        super(Linear_trans, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_dim)


    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x
