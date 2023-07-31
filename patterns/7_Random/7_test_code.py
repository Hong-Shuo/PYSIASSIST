import torch
from torch import nn
import os
import numpy as np
import random

torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(3)
random.seed(3)
os.environ['PYTHONHASHSEED'] = str(3)

linear = nn.Linear(5, 2)
linear2 = nn.Linear(5, 2)

print(linear.weight)
print(linear2.weight)