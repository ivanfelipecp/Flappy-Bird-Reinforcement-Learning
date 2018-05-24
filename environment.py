import sys
import image_functions
import random
import gym
import gym_ple
from episode import Episode

env = gym.make('FlappyBird-v0' if len(sys.argv)<2 else sys.argv[1])

# Agente
#agent = RandomAgent(env.action_space)

episode_count = 100
reward = 0
done = False
cont = 0
ep = Episode()

for i in range(episode_count):
    # Agregarla
    ob = env.reset()
    ep.reset()

    while True:
        cont += 1

        # 0 va hacia arriba, 1 para abajo
        #agent.act(ob, reward, done) 
        action = random.randint(0,1)
        ob, reward, done, _ = env.step(action) # reward -5 y done true cuando pierde
        if done:
            break

        #ob, action, reward
        ep.add_info(ob, action, reward)
        #img = image_functions.ob_2_gray(ob)
        #image_functions.save_image(img,"save"+str(cont))


        #env.render()
    ep.do()
    input("waiting...")

env.close()