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

    def forward(self, hn, xn, x):

        
        return hn, xn










        

        



