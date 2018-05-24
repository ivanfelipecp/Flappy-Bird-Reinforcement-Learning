import numpy as np
import image_functions
import sys
import cv2

class Episode():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add_info(self, ob, action, reward):
        self.states += [image_functions.ob_2_gray(ob)]
        self.actions += [[1,0] if action == 0 else [0,1]]
        self.rewards += [reward]

    def do(self):
        self.states = self.new_states()
        self.rewards = self.new_rewards()
        
        print(self.rewards)
        sys.exit("adansito")

    def new_states(self):
        new_states = []
        m = len(self.states)
        for i in range(1,m):
            new_states += [np.abs(self.states[i]-self.states[i-1]).astype(np.uint8)]
            image_functions.save_image(new_states[-1],"adancito"+str(i))
        return new_states
    
    def new_rewards(self):
        positive_reward = 1
        negative_reward = -1
        new_rewards = []
        flag = False
        m = len(self.rewards)
        for i in reversed(range(m)):
            if self.rewards[i] == 1:
                flag = True
            new_rewards = [positive_reward if flag else negative_reward] + new_rewards

        return new_rewards