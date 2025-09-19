import torch

def l1_weight(rnn, scale):
    l1 = 0
    for name, param in rnn.named_parameters():
        l1 += torch.mean(torch.abs(torch.flatten(param)))
    l1 *= scale
    return l1

def l1_rate(act, scale):
    l1 = scale * torch.mean(torch.abs(torch.flatten(act)))
    return l1