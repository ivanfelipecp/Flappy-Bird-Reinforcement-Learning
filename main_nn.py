import argparse
import sys
import gym
import gym_ple
import numpy as np
from itertools import count
from collections import namedtuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import image_functions

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
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

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(10000, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.drop1(self.affine1(x)))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

# Declaramos la red y el optimizador
model = Policy()
try:
    model.load_state_dict(torch.load(weights_file))
except:
    print("No se pudieron cargar los pesos")
optimizer = optim.Adam(model.parameters(), lr=0.0085)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def change_rewards():
    positive_reward = 1
    negative_reward = -1
    flag = False
    m = len(model.rewards)
    for i in reversed(range(m)):
        if model.rewards[i] > 0:
            flag = True
        model.rewards[i] = positive_reward if flag else negative_reward

def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    change_rewards()
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

episodes = 1000
def main():
    for i_episode in range(episodes):
        print("Episode", i_episode)
        last_state = env.reset()
        while True:  # Don't infinite loop while learning
            gray = image_functions.ob_2_gray(last_state).ravel()
            action = select_action(gray)
            state, reward, done, _ = env.step(action)
            if done:
                break
            model.rewards.append(reward)
            last_state = np.abs(state - last_state)
            env.render()
            

        finish_episode()
    torch.save(model.state_dict(), weights_file)


if __name__ == '__main__':
    main()
