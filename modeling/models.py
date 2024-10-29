import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class mRNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(mRNN, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.bla2acc = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.bla2bla = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.acc2bla = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.acc2acc = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.inp_weight = nn.Parameter(torch.empty(size=(hid_dim, inp_dim)))
        self.bias = nn.Parameter(torch.empty(hid_dim,))

        # Weight Initialization
        nn.init.uniform_(self.bla2acc, 0, 1e-1)
        nn.init.uniform_(self.bla2bla, 0, 1e-1)
        nn.init.uniform_(self.acc2bla, 0, 1e-1)
        nn.init.uniform_(self.acc2acc, 0, 1e-1)
        nn.init.uniform_(self.inp_weight, 0, 1e-1)
        nn.init.uniform_(self.bias, 0, 1e-1)

        # Sparsity Masks
        sparse_matrix = torch.ones(size=(hid_dim, hid_dim))
        nn.init.sparse_(sparse_matrix, 0.5)

        # Dales Law Exc / Inhib
        self.sign_matrix = torch.eye(hid_dim)
        self.sign_matrix[:, int(hid_dim / 2):] *= -1

        self.tau = 0.1

    def forward(self, hn, xn, x):

        size = x.shape[1]
        hn_next = hn
        xn_next = xn
        hn_list = []
        xn_list = []

        bla2acc = (self.sparse_matrix * F.relu(self.bla2acc)) @ self.sign_matrix
        bla2bla = (self.sparse_matrix * F.relu(self.bla2bla)) @ self.sign_matrix
        acc2bla = (self.sparse_matrix * F.relu(self.acc2bla)) @ self.sign_matrix
        acc2acc = (self.sparse_matrix * F.relu(self.acc2acc)) @ self.sign_matrix
        inp_weight = F.relu(self.inp_weight) @ self.sign_matrix
        bias = F.relu(self.bias)

        W_bla = torch.cat([bla2bla, acc2bla], dim=1)
        W_acc = torch.cat([bla2acc, acc2acc], dim=1)
        W_rec = torch.cat([W_bla, W_acc])

        for t in range(size):

            xn_next = (
                xn_next 
                * self.tau 
                * (
                    -xn_next
                    + (W_rec @ hn_next.T).T
                    + (inp_weight @ x[:, t, :].T).T
                    + bias
                )
            )

            hn_next = F.relu(xn_next)

            xn_list.append(xn_next)
            hn_list.append(hn_next)
        
        hn = torch.stack(hn_list, dim=1)
        xn = torch.stack(xn_list, dim=1)
        
        return hn, xn










        

        



