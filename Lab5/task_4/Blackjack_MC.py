import gym
import numpy as np
from visualizeValue import visualizeValue
env = gym.make('Blackjack-v0')

epochs = 100000 # Number of episodes/plays
epsilon = 1. # E-greedy
gamma = 1.0 # Discount factor 

############# SOLUTION AS FOLLOWS ####################################################
alpha = 0.1

# Initialize the Q table
# N.B. The state is in fact 21 * 10 * 2 scenarios (Current score * Card value * Usable Ace or Not), there are 2 actions
Q = np.zeros((21,10,2,2)) 

# Initialize the policy
def policy(s):
    '''
        Make soft policy $\pi(s,a)$
    '''
    p_a = np.ones(2, dtype=float) * epsilon / 2
    a_max = np.argmax(Q[s[0]-1, s[1]-1, int(s[2]),:])
    p_a[a_max] += (1.0 - epsilon)
    return p_a

# Initialize the sets
R_sum = np.zeros((21,10,2,2))
R_cnt = np.zeros((21,10,2,2))

# For each episode (i.e., each game)
for i in range(epochs):

    # Initialize new game/episode (choose and observe a random initial state)
    episode = []        
    s  = env.reset() 

    # Generate the episode (using e-soft policy)
    done = False
    while not done:

        # Act with the epsilon-soft policy
        p_a = policy(s)
        a = np.random.choice(np.arange(2), p=p_a)

        # Take the action, get the reward and next state
        s_, r, done, _ = env.step(a)
        # ... and save the (s, a, r) tuple
        episode.append((s, a, r))

        if done:
            break

        # Set the next state
        s = s_

    # For each (s,a) pair in the episode
    sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
    for s,a in sa_in_episode:
        sa_pair = (s, a)
        # Find the first occurence         
        first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == s and x[1] == a)
        # Calculate the return 
        R = sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occurence_idx:])])
        # Store in the set
        R_sum[sa_pair] += R
        R_cnt[sa_pair] += 1
        # Policy improvement
        #Q[s][a] = R_sum[sa_pair] / R_cnt[sa_pair]
        Q[s[0]-1, s[1]-1, int(s[2]),a] = R_sum[sa_pair] / R_cnt[sa_pair]

    # Decay epsilon
    #epsilon = epsilon*0.9999
   
    if i > 0 and (i % 1000) == 0:
        print("Episode: ", i, "Average Return: ", R_sum/i)
    
printQ(Q) 

visualizeValue(Q)

