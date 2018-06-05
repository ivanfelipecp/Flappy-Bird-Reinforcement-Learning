import argparse
import gym
import gym_ple
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import image_functions

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--weights', type=str, default="weights", metavar='Name',
                    help='Pesos de la red')        
args = parser.parse_args()


env = gym.make('FlappyBird-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
weights_file = args.weights

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.h1 = nn.Linear(10000, 128)
        self.h2 = nn.Linear(128, 64)
        self.h3 = nn.Linear(64, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        action_scores = self.h3(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
try:
    policy.load_state_dict(torch.load(weights_file))
except:
    print("No se pudieron cargar los pesos")
optimizer = optim.SGD(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()

def change_rewards():
    positive_reward = 1
    negative_reward = -1
    flag = False
    m = len(policy.rewards)
    for i in reversed(range(m)):
        if policy.rewards[i] > 0:
            flag = True
        policy.rewards[i] = positive_reward if flag else negative_reward

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    change_rewards()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

episodes = 100
def main():
    for i_episode in range(episodes):
        print("Episode", i_episode)
        state = env.reset()
        while True:  # Don't infinite loop while learning
            gray = image_functions.ob_2_gray(state).ravel()
            action = select_action(gray)
            state, reward, done, _ = env.step(action)
            if done:
                break
            
            policy.rewards.append(reward)
            env.render()
        finish_episode()
    torch.save(policy.state_dict(), weights_file)


if __name__ == '__main__':
    main()