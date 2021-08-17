
import numpy as np
import time
import torch
from torch import nn, optim
import sys
sys.path.append("..")


x = torch.tensor(2.0,requires_grad = True)
y=x**2
y.backward()
print(x.grad)
with torch.no_grad():
    x.grad.zero_()
print(x.grad)

