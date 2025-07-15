import numpy as np
import torch

from utils import choose_nonlinearity


class MLP(torch.nn.Module):
    '''Just a MLP'''

    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh', bias_bool=True):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=bias_bool)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x, separate_fields=False):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)




if __name__ == "__main__":
    x = torch.randn(6)
    md = MLP1(input_dim=6, output_dim=1)
    res = md(x)
    print(res.shape)
