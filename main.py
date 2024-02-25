import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import yfinance as yf
import matplotlib.pyplot as plt
from library import *

NUM_EPISODES = 1

# Hyperparameters
n = 1  # look back period
input_dim = 6 # OHLCV and Adjusted Close in yahoo finance Data
hidden_dim = 64
action_set = [-1,0,1]
output_dim = len(action_set)  # buy, sell, do nothing for now
epsilon = 0.98
epsilon_decay = 0.995
gamma = 0.65
learning_rate = 0.035
max_size_replay = 1000
N = deque(maxlen=max_size_replay) # replay memory size
batch_size = 100
C = 300  # number of steps to restart weights of the target network
tw = 5  # time window of action stabilization

# Initialize environment
env = StockTradingEnv(ticker='AI.PA', start='2020-10-01', end='2023-11-22', lookback=n,tw=tw)
if input_dim != env.data.shape[1]:
    print('input_dim and env data shape conflict !') # OHLCV and Adjusted Close in yahoo finance Data
    input_dim = env.data.shape[1]
T = env.data.shape[0]
print('T:',T)

# Initialize DQN and target network
Q = DQN(input_dim, hidden_dim, output_dim)
Q_target = DQN(input_dim, hidden_dim, output_dim)
Q_target.load_state_dict(Q.state_dict()) # theta_target = theta at time 0 (and every C steps)
optimizer = optim.Adam(Q.parameters(),lr=learning_rate)
REWARDS = []

# Sampling and Training 
for episode in range(NUM_EPISODES):
    # Initialize environment
    state = env.reset()
    N.clear()
    REWARDS.clear()
    state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    w = 0
    for t in range(n,T-tw):
        print('TIMESTEP:', t)
        if w == 0:
            if random.random() < epsilon:
                action = random.choice(action_set)
                print('selecting random action:',action)
                w = tw
            else:
                action = torch.argmax(Q(state)).item()-1 #-1 to make our python indexing compatible with the action set
                #print('action in policy:',action)

        # Perform action and get new state and reward
        new_state, reward,done = env.step(action)
        new_state = torch.tensor(np.array(new_state), dtype=torch.float32).unsqueeze(0)

        # Store experience in replay memory
        N.append((state, action, reward, new_state,done))
        if len(N) > max_size_replay: # if replay memory is full, remove the oldest experience
            N.pop(0)
        REWARDS.append(reward)

        # Sample random minibatch from replay memory
        minibatch = random.sample(N, min(len(N), batch_size)) #sample random minibatch from replay memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)   #(s_t, a_t, r_t, s_{t+1}, done) 

        # EXPERIENCE : Random states as input to check if Q values are getting updated correctly
        # Plug in random values instead of using actual state and next state batches to test the code
        state_batch = [torch.randn_like(state) for state in state_batch]
        next_state_batch = [torch.randn_like(next_state) for next_state in next_state_batch]
        
        # Compute Q values for current states with Q-network
        Q_values = Q(torch.cat(state_batch))  # Compute Q values for current states with Q-network
        next_Q_values = Q_target(torch.cat(next_state_batch))  # Compute Q values for next states with target network
        max_next_Q_values, _ = next_Q_values.max(dim=2) # Compute max Q values for next states with target network (because Bellman needs max!)

        #Compute the target Q values
        targets = np.array(reward_batch) + gamma*np.squeeze(max_next_Q_values.detach().numpy()) # y_i = r_i + gamma*max_a' Q_target(s',a')
        print('max_next_Q_values',max_next_Q_values,max_next_Q_values.shape)
        Q_tmp = np.squeeze(Q_values.detach().numpy())
        Q_values_batch = []
        if t == n:
            Q_values_batch = Q_tmp[action_batch[0]+1]
        else:
            for i in range(Q_tmp.shape[0]):
                Q_values_batch.append(Q_tmp[i][action_batch[i]+1]) # only works if action_set = [-1,0,1,..,k]

        # Compute and minimize the MSE loss
        loss = nn.functional.mse_loss(torch.tensor(Q_values_batch,dtype=torch.float32), torch.tensor(targets,dtype=torch.float32))
        loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward() # Compute the gradient and update the weights
        optimizer.step()

        if t % C == 0:
            Q_target.load_state_dict(Q.state_dict())  #Reset Q_target to Q every C steps
            print('Reset target weights - C step')

        # Updates #
        epsilon = epsilon_decay * epsilon
        state = new_state
        w = w-1

        if done:
            break

        print('\n')


print('\n')
print('Training complete')
print('\n')
# for param in Q.parameters():
#     print(param,param.data)

plt.plot(REWARDS,marker='o')
plt.title('Training progress of DQN')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

plt.hist(REWARDS, bins=30)
# add a vertical line with the mean of rewards to the histogram
plt.axvline(np.mean(REWARDS), color='k', linestyle='dashed', linewidth=1,label='Mean reward')
plt.title('Histogram of rewards')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# add a plot of the evolution of some weights of the Q-network
# for param in Q.parameters():
#     plt.plot(param.data)
#     plt.show()