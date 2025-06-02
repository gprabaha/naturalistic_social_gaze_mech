import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mRNNTorch.mRNN import mRNN
from mRNNTorch.utils import get_region_activity

class Model(nn.Module):
    def __init__(self, 
                 config, 
                 hid_dim, 
                 pfc_units, 
                 acc_units, 
                 ofc_units, 
                 bla_units, 
                 dt, 
                 tau, 
                 inp_noise, 
                 act_noise, 
                 constrained, 
                 batch_first, 
                 spectral_radius
        ):
        super(Model, self).__init__()

        self.hid_dim = hid_dim
        self.dt = dt
        self.tau = tau
        self.inp_noise = inp_noise
        self.act_noise = act_noise

        self.mrnn = mRNN(
            config,
            constrained=constrained,
            batch_first=batch_first,
            dt=dt,
            tau=tau,
            noise_level_act=act_noise,
            noise_level_inp=inp_noise,
            spectral_radius=spectral_radius
        )

        self.connection_props = [
            {"name": "pfc_exc",
             "sign": "exc"},
            {"name": "pfc_inhib",
             "sign": "inhib"},
            {"name": "acc_exc",
             "sign": "exc"},
            {"name": "acc_inhib",
             "sign": "inhib"},
            {"name": "ofc_exc",
             "sign": "exc"},
            {"name": "ofc_inhib",
             "sign": "inhib"},
            {"name": "bla_exc",
             "sign": "exc"},
            {"name": "bla_inhib",
             "sign": "inhib"}
        ]

        # Build fully connected network with proper cell types
        for src_region in self.connection_props:
            for dst_region in self.connection_props:
                self.mrnn.add_recurrent_connection(src_region["name"], dst_region["name"], sign=src_region["sign"])
        self.mrnn.finalize_connectivity()

        self.pfc_out = nn.Linear(hid_dim, pfc_units)
        self.acc_out = nn.Linear(hid_dim, acc_units)
        self.ofc_out = nn.Linear(hid_dim, ofc_units)
        self.bla_out = nn.Linear(hid_dim, bla_units)

    def forward(self, xn, inp, *args, noise=True):
        xn, hn = self.mrnn(xn, inp, *args, noise=noise)

        pfc_act = get_region_activity(self.mrnn, hn, "pfc_exc", "pfc_inhib")
        acc_act = get_region_activity(self.mrnn, hn, "acc_exc", "acc_inhib")
        ofc_act = get_region_activity(self.mrnn, hn, "ofc_exc", "ofc_inhib")
        bla_act = get_region_activity(self.mrnn, hn, "bla_exc", "ofc_inhib")

        pfc_out = self.pfc_out(pfc_act)
        acc_out = self.acc_out(acc_act)
        ofc_out = self.ofc_out(ofc_act)
        bla_out = self.bla_out(bla_act)

        out = torch.cat([ofc_out, bla_out, pfc_out, acc_out], dim=-1)

        return out, hn