import sys
import image_functions
import random
import gym
import gym_ple
import numpy as np
import image_functions
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import image_functions
from torch.autograd import Variable
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(10000, 256)
        self.fc2 = nn.Linear(256, 2)
        #self.fc = nn.Linear(64, 2)

        self.saved_log_probs = []
        self.rewards = []
        self.states = []

    def forward(self, x):
        a = self.conv1(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        print("lego")
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 10000)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = 1)

    def pre_end(self):
        positive_reward = 5
        negative_reward = -1
        new_rewards = []
        flag = False
        m = len(self.rewards)
        for i in reversed(range(m)):
            if self.rewards[i] > 0:
                flag = True
            new_rewards = [positive_reward if flag else negative_reward] + new_rewards

        self.rewards = new_rewards        

# Declaramos la red y el optimizador
policy = Policy()
try:
    policy.load_state_dict(torch.load('weights'))
except:
    print("No se pudieron cargar los pesos")
optimizer = optim.SGD(policy.parameters(), lr=0.0085, momentum=0.9)
eps = np.finfo(np.float32).eps.item()

# Funciones aca

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    # Hace forwards para obtener dos resultados
    probs = policy(state)
    # Esto es para escoger la probabilidad más alta
    m = Categorical(probs)
    # Escoje la más alta
    action = m.sample()
    # Guarda el log de la acción para hacer train
    policy.saved_log_probs.append(m.log_prob(action))
    # Retorna la acción
    return action.item()

def end_of_episode():
    gamma = 0.99
    policy_loss = []
    rewards = []
    discount_reward = 0
    flag = True
    for r in policy.rewards[::-1]:
        if r > 0 and flag:
            discount_reward = 0
            flag = False
        discount_reward = r + gamma * discount_reward
        rewards.insert(0, discount_reward)
    
    #Convierte los rewards a tensor
    rewards = torch.tensor(rewards)
    # Normaliza el tensor
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    # Saca el policy loss
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return policy_loss.item()
# Fin Funciones


# Ambiente

env = gym.make('FlappyBird-v0' if len(sys.argv)<2 else sys.argv[1])
#print(env.spec.reward_threshold)
episode_count = 10000
for i in range(episode_count):
    # Agregarla
    last_ob = env.reset()
    #print("Inicio del episodio",i)
    while True:
        # 0 va hacia arriba, 1 para abajo
        #agent.act(ob, reward, done)
        last_ob_gray = image_functions.ob_2_gray(last_ob).ravel()        
        action = select_action(last_ob_gray)#random.randint(0,1) #policy(last_ob)
        ob, reward, done, _ = env.step(action) # reward -5 y done true cuando pierde
        if done:
            break

        policy.states.append(last_ob_gray)
        policy.rewards.append(reward)
        last_ob = np.abs(ob - last_ob)
        env.render()
    
    policy.pre_end()
    loss = end_of_episode()
    #print("Loss del episodio",i,"->",loss)
    #print("Fin del episodio",i)

env.close()

torch.save(policy.state_dict(), "weights")
#p.load_state_dict(torch.load('pesos'))
#save_files.save(p.state_dict, "pesos")