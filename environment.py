import sys

import gym
from gym.wrappers import Monitor
import gym_ple
import image_functions

env = gym.make('FlappyBird-v0' if len(sys.argv)<2 else sys.argv[1])

# Agente
#agent = RandomAgent(env.action_space)

episode_count = 100
reward = 0
done = False
cont = 0

for i in range(episode_count):
    ob = env.reset()

    while True:
        cont += 1

        # 0 va hacia arriba, 1 para abajo
        #agent.act(ob, reward, done) 
        action = 0
        ob, reward, done, _ = env.step(action) # reward -5 y done true cuando pierde
        if done:
            break

        # img = image_functions.ob_2_gray(ob)
        # image_functions.save_image(img,"save"+str(cont))

        #env.render()

env.close()