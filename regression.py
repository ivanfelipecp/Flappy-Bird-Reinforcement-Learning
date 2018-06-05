import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import image_functions


from ple.games.flappybird import FlappyBird
from ple import PLE

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    print(probs)
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
    gamma = 0.99
    R = 0
    policy_loss = []
    rewards = []
    change_rewards()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
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



game = FlappyBird()
game.actions = {"up":1}
p = PLE(game, fps=30, display_screen=True)

p.init()
for i in range(1000):
    p.reset_game()
    while not p.game_over():
        #observation = p.getScreenRGB()
        info = game.getGameState()
        data = np.array([
            info['player_y'],
            info['next_pipe_dist_to_player'],
            info['next_pipe_top_y'],
            info['next_pipe_bottom_y']
        ])
        #data = (data - np.min(data))/ (np.max(data) - np.min(data))
        #print(data)
        #input()
        action = select_action(data)
        policy.rewards.append(p.act(action))
    finish_episode()