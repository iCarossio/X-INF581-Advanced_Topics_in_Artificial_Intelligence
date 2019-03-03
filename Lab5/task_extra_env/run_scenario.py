import gym
import gym_myenv

env = gym.make('myenv-v0')
env = env.unwrapped

from basic_agent import StupidAgent
agent = StupidAgent(env.action_space)

for i_episode in range(20):
    print("Episode ", (i_episode+1))
    reward = 0
    done = False
    observation = env.reset()
    for t in range(10):
        env.render()
        action = agent.act(observation, reward, done)
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

