import gym
env = gym.make('FrozenLake-v0')

#n_a = env.action_space.n
#n_s = env.observation_space.n

#print("Nb_Actions: {}".format(n_a))
#print("Nb_States: {}".format(n_s))

#Todo: Define a Q-table to host the Q-function estimate
q_table = np.zeros(()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample() # Returns a random action.
        # Todo: Replace the above function with one that selects an action based on your algorithm (i.e., SARSA with $\epsilon$-greedy exploration.
        
        observation, reward, done, info = env.step(action)
        # Todo: Update the tabular estimate of the Q-function using the update rule of the algorithm(SARSA or Q-learning)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
