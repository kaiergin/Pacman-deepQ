import gym
from MsPacmanDeepQ import MsPacman
import numpy as np

IM_WIDTH = 160
IM_HEIGHT = 210

# Create the environment
env = gym.make('MsPacman-v0', frameskip=1)
# Create the agent
pac = MsPacman(epsilon=.5)

NUM_STEPS = 4

RENDER = False
num_games = 20000

for i in range(num_games):
    obs = env.reset()
    obs = [obs for _ in range(NUM_STEPS)]
    action = pac.eval_policy(obs, 0, 1)
    rewards = []
    while True:
        obs, reward = [], 0
        done = False
        for _ in range(NUM_STEPS):
            o, r, d, _ = env.step(action)
            obs.append(o)
            reward += r
            done = d
            not_done = 0 if d else 1
        if RENDER:
            env.render()
        rewards.append(reward)
        if done:
            action = pac.eval_policy(obs, -10, not_done)
            pac.fit_policy()
            cum_rewards = np.sum(rewards)
            print('Game:', i, 'Cumulative reward:', cum_rewards)
            break
        else:
            action = pac.eval_policy(obs, reward, not_done)