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
        # Resta la observación actual con la anterior
        
        self.states += [image_functions.ob_2_gray(new_ob).ravel()]
        self.actions += [[1,0] if action == 0 else [0,1]]
        self.rewards += [reward]

    def pre_finish(self):
        self.rewards = self.set_rewards()
        #print(self.rewards)
        sys.exit("cya")
    
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