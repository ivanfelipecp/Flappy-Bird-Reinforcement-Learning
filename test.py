import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.distributions import Categorical

"""
state= state.reshape(1, 1, state.shape[0], state.shape[1])
#Convierte los valores a float, asi lo recibe conv2d
state = state.type(torch.FloatTensor)


"""
a = Variable(torch.tensor(np.array([1,3,5,6]))
values, indices = torch.max(a, 1)
print(values)
print(indices)