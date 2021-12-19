import torch

# v = [torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3)]
# a = [torch.unsqueeze(v[i], dim=1) for i in range(len(v))]
# for idx in range(len(v)):
#     v[idx] = torch.unsqueeze(v[idx], dim=1)

# c = torch.unsqueeze(v1, 1)
# a = torch.cat((v1, v2), dim=0)
# b = torch.cat((v1, v2), dim=1)
# res = torch.Tensor(2, 3, 3)
# b = torch.Tensor()
#
# torch.cat(a, out=b, dim=1)
# a = torch.randn(4, 4, 4)
# b, c = torch.median(a, 1)
# print(int(40 / 7))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

a = torch.randn([2, 4, 3])
# b = torch.randn([2, 4, 3])
# c = torch.randn([2, 4, 3])
# idx = [3, 1,2,0]
# # torch.gather(a, dim=1, index=torch.tensor())
# b = a[:,idx]
# b = torch.transpose(a, 0, 1)
# res = torch.cat([a,b,c], dim=0)
res = torch.flatten(a, start_dim=0, end_dim=0)

pass

