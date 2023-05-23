import time

import retro

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

obs = env.reset()
done = False
rewards = 0
while not done:
    env.render()
    obs, reward, done, info = env.step(env.action_space.sample())
    if reward>0:
        rewards+=reward
    if done:
        print(rewards)