import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import save_files

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # Entrada -> batch_size x 1000
        self.h1 = nn.Linear(100,10)
        self.h2 = nn.Linear(10,1)
        self.h3 = nn.Linear(1,2)

        self.saved_log_probs = []
        self.rewards = []


    def forward(self, x):
        x = Variable(torch.Tensor(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        return F.softmax(x, dim=1)

    def set_rewards(self):
        positive_reward = 1
        negative_reward = -1
        new_rewards = []
        flag = False
        m = len(self.rewards)
        for i in reversed(range(m)):
            if self.rewards[i] == 1:
                flag = True
            new_rewards = [positive_reward if flag else negative_reward] + new_rewards

        self.rewards = new_rewards
p = Policy()
#torch.save(p.state_dict(), "pesos")
#p.load_state_dict(torch.load('pesos'))
#save_files.save(p.state_dict, "pesos")

#x = np.array([[2]*100])
#print(p(x))