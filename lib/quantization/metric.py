import torch

def sqnr(a,b):
    dtype = a.dtype
    a = a.data.float()
    b = b.data.float()
    if (a-b).sum() == 0: # all input is same
        return float('inf') # infinite
    else:
        s_power = 0
        qn_power = 0
        # torch implementation
        s_power = torch.sum(torch.pow(a,2))
        qn_power = torch.sum(torch.pow(a-b,2))
        sqnr = 10.0*torch.log10(s_power/qn_power)
        return sqnr.item()

def mse(a,b):
    return (a-b).pow(2).mean().item()

def kurtosis(x):
    x = x.reshape(-1).float()
    s1 = (x-x.mean()).pow(4).mean()
    s2 = (x-x.mean()).pow(2).mean().pow(2)
    return (s1/s2-3).item()

# For test
if __name__=='__main__':
    x = torch.randn([10000])
    print(kurtosis(x))
    import numpy as np
    from scipy.stats import kurtosis as npkurtosis
    print(npkurtosis(x.numpy()))
