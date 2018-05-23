import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class XorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

m = XorNet()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(m.parameters(), lr=1e-3)

training_epochs = 3000
minibatch_size = 32

# input-output pairs
pairs = [(np.asarray([0.0,0.0]), [0.0]),
         (np.asarray([0.0,1.0]), [1.0]),
         (np.asarray([1.0,0.0]), [1.0]),
         (np.asarray([1.0,1.0]), [0.0])]

state_matrix = np.vstack([x[0] for x in pairs])
label_matrix = np.vstack([x[1] for x in pairs])

for i in range(training_epochs):
        
    for batch_ind in range(4):
        # wrap the data in variables
        minibatch_state_var = Variable(torch.Tensor(state_matrix))
        minibatch_label_var = Variable(torch.Tensor(label_matrix))
                
        # forward pass
        y_pred = m(minibatch_state_var)
        
        # compute and print loss
        loss = loss_fn(y_pred, minibatch_label_var)
        #print(i, batch_ind, loss.data[0])

        # reset gradients
        optimizer.zero_grad()
        
        # backwards pass
        loss.backward()
        
        # step the optimizer - update the weights
        optimizer.step()

print("Function after training:")
print("f(0,0) = {}".format(m(Variable(torch.Tensor([0.0,0.0]).unsqueeze(0)))))
print("f(0,1) = {}".format(m(Variable(torch.Tensor([0.0,1.0]).unsqueeze(0)))))
print("f(1,0) = {}".format(m(Variable(torch.Tensor([1.0,0.0]).unsqueeze(0)))))
print("f(1,1) = {}".format(m(Variable(torch.Tensor([1.0,1.0]).unsqueeze(0)))))

