import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import yfinance as yf
from library import *

NUM_EPISODES = 5
TICKER = 'AI.PA' # Ticker of the stock you want to train the algorithm on
START_DATE = '2018-06-01' # Training Data Start Date
END_DATE = '2023-03-22' # Training Data End Date
OOS_START_DATE = '2023-03-28' # Out of Sample Data Start Date
OOS_END_DATE = '2024-03-01' # Out of Sample Data End Date
ACTION_SET = [-1,0,1] # sell, hold, buy

REWARDS = []
LOSSES = []
MEAN_LOSSES = []
ACTIONS_OOS = []

# Hyperparameters
n = 1  # look back period

input_dim = 6 # OHLCV and Adjusted Close in yahoo finance Data
hidden_dim = 64
output_dim = len(ACTION_SET)  # buy, sell, do nothing for now

epsilon = 0.80
epsilon_decay = 0.95
gamma = 0.975
learning_rate = 0.05
max_size_replay = 1000
N = deque(maxlen=max_size_replay) # replay memory size
batch_size = 200
C = 120  # number of steps to restart weights of the target network
tw = 5  # time window of action stabilization
tf = '1d' # time frame of the data

# Initialize environment
env = StockTradingEnv(ticker=TICKER, start=START_DATE, end=END_DATE, lookback=n,tw=tw,tf=tf)
if input_dim != env.data.shape[1]:
    print('input_dim and env data shape conflict !') # OHLCV and Adjusted Close in yahoo finance Data
    input_dim = env.data.shape[1]
T = env.data.shape[0]
print('T:',T)
print('env data',env.data.head(3))

# Initialize DQN and target network
Q = DQN(input_dim, hidden_dim, output_dim)
Q_target = DQN(input_dim, hidden_dim, output_dim)
Q_target.load_state_dict(Q.state_dict()) # theta_target = theta at time 0 (and every C steps)
#optimizer = optim.Adam(Q.parameters(),lr=learning_rate)
optimizer = optim.Adam(Q.parameters())

# Sampling and Training 
for episode in range(NUM_EPISODES):
    # Initialize environment
    state = env.reset()
    N.clear()
    REWARDS.clear()
    LOSSES.clear()
    state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    w = 0
    for t in range(n,T-tw):
        if w == 0:
            if random.random() < epsilon:
                action = random.choice(ACTION_SET)
                print('selecting random action:',action)
                w = tw
            else:
                action = torch.argmax(Q(state)).item()-1 #-1 to make our python indexing compatible with the action set

        # Perform action and get new state and reward
        new_state, reward, done = env.step(action)
        new_state = torch.tensor(np.array(new_state), dtype=torch.float32).unsqueeze(0)

        # Store experience in replay memory
        N.append((state, action, reward, new_state, done))
        if len(N) > max_size_replay:  # if replay memory is full, remove the oldest experience
            N.popleft()  # Use popleft() instead of pop(0) for deque
        REWARDS.append(reward)
        
        # Sample random minibatch from replay memory
        minibatch = random.sample(N, min(len(N), batch_size))  # sample random minibatch from replay memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)  # (s_t, a_t, r_t, s_{t+1}, done)

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
        LOSSES.append(loss.item())
        # add condition to add the loss to the list only if it is at the end of the episode
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
    
    mean_loss = np.mean(LOSSES)
    MEAN_LOSSES.append(mean_loss)

print('\n')
print('Training complete')
print('\n')

# Plot the rewards and the mean reward histogram
plt.hist(REWARDS, bins=30)
# add a vertical line with the mean of rewards to the histogram
plt.axvline(np.mean(REWARDS), color='k', linestyle='dashed', linewidth=1,label='Mean reward')
plt.title('Histogram of rewards')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Plot the losses
plt.plot(MEAN_LOSSES, marker='o')
plt.title('Mean Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

print('Saving the model ...')
torch.save(Q.state_dict(), 'Q.pt')
print('Launching Out of Sample Testing  ...')
print('\n')

# Out of Sample Testing
data_bis = yf.download(TICKER,OOS_START_DATE, OOS_END_DATE)

data_bis = (data_bis - env.data.mean()) / env.data.std() # standardize with data in sample to stay consistant
#data_bis = (data_bis - data_bis.mean()) / data_bis.std() # uncomment standardize with data oos

# epsilon greedy for action selection in OOS as an example
epsilon_oos = 0.05
for t in range(n,len(data_bis)-tw):
    state = data_bis.iloc[t-n:t]
    state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    if random.random() < epsilon_oos:
        action = random.choice(ACTION_SET)
    else:
        action = torch.argmax(Q(state)).item()-1
    ACTIONS_OOS.append(action)

print('ACTIONS:',ACTIONS_OOS)

# Plot the actions and the Close Price to visualize the strategy
data_bis['Close'].plot()
# if action == 1 I want to have a triangle pointing up else a triangle pointing down
for i in range(len(ACTIONS_OOS)):
    if ACTIONS_OOS[i] == 1:
        plt.plot(data_bis.index[i],data_bis['Close'].iloc[i],marker='o',color='navy',label='Buy')
    elif ACTIONS_OOS[i] == -1:
        plt.plot(data_bis.index[i],data_bis['Close'].iloc[i],marker='v',color='orchid',label='Sell')
    else:
        pass

plt.xlabel('Time')
plt.ylabel(f'Standardized Close Price of {TICKER}')
plt.title('Out of Sample Actions')
# Get the current axes
ax = plt.gca()
# Handle duplicates in the legend
handles, labels = ax.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
ax.legend(*zip(*unique))
plt.show()
