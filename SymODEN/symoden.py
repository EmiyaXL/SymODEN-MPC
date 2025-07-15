import torch
import numpy as np

from nn_models import MLP


class SymODEN_R(torch.nn.Module):
    '''
    Architecture for input (q, p, u), 
    where q and p are tensors of size (bs, n) and u is a tensor of size (bs, 1)
    '''
    def __init__(self, input_dim, H_net=None, g_net=None, device=None,
                    assume_canonical_coords=True):
        super(SymODEN_R, self).__init__()
        self.u = 0
        self.H_net = H_net
        self.g_net = g_net

        self.device = device
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)
        self.nfe = 0
        self.input_dim = input_dim

    def get_u(self, u):
        self.u = u

    def forward(self, t, x):
        with torch.enable_grad():
            one = torch.tensor(1, dtype=torch.float32, device=self.device, requires_grad=True)
            x = one * x
            self.nfe += 1
            q, p = torch.chunk(x, 2, dim=1)
            q_p = torch.cat((q, p), dim=1)
            H = self.H_net(q_p)

            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            H_vector_field = torch.matmul(dH, self.M.t())
            g_q = self.g_net(q)

            F = g_q * u
            F_vector_field = torch.cat((torch.zeros_like(F), F), dim=1)

            return H_vector_field + F_vector_field