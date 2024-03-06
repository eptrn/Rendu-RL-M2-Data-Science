import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import pandas as pd
from library import *

NUM_EPISODES = 25
TICKER = 'AI.PA' # Ticker of the stock you want to train the algorithm on
START_DATE = '2019-01-01' # Training Data Start Date
END_DATE = '2023-03-22' # Training Data End Date
OOS_START_DATE = '2022-03-28' # Out of Sample Data Start Date
OOS_END_DATE = '2022-11-01' # Out of Sample Data End Date
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
tw = 5  # time window of action stabilization
tf = '1d' # time frame of the data
epsilon = 0.99
epsilon_decay = 0.985
gamma = 0.875
learning_rate = 0.05
max_size_replay = 1000
N = deque(maxlen=max_size_replay) # replay memory size
batch_size = 64
C = 80  # number of steps to restart weights of the target network

# Initialize environment
env = StockTradingEnv(ticker=TICKER, start=START_DATE, end=END_DATE, lookback=n,tw=tw,tf=tf)
#do a min max normalization, store the min and max, useful later for out of sample normalisation
ohlcv_mean = env.data.mean()
ohlcv_std = env.data.std()
env.data = (env.data - ohlcv_mean) / ohlcv_std
if input_dim != env.data.shape[1]:
    print('input_dim and env data shape conflict !') # OHLCV and Adjusted Close in yahoo finance Data
    input_dim = env.data.shape[1]
T = env.data.shape[0]
print('T = ',T)

#Grid Search for the best parameters
C_list = [70,90,125]
batch_size_list = [64,128,256] # Please write only integers
gamma_list = [0.5,0.75,0.95]
PARAM_LOSS = []

# Sampling and Training 
for C in C_list:
    for batch_size in batch_size_list:
        for gamma in gamma_list:
            # Initialize DQN and target network
            Q = DQN(input_dim, hidden_dim, output_dim)
            Q_target = DQN(input_dim, hidden_dim, output_dim)
            Q_target.load_state_dict(Q.state_dict()) # theta_target = theta at time 0 (and every C steps)
            #optimizer = optim.Adam(Q.parameters(),lr=learning_rate) # uncomment to use a custom learning rate
            optimizer = optim.Adam(Q.parameters())
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
                    #print('max_next_Q_values',max_next_Q_values,max_next_Q_values.shape) #uncomment to check the values are not constant 
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
                    optimizer.zero_grad()
                    loss.backward() # Compute the gradient and update the weights
                    optimizer.step()

                    if t % C == 0:
                        Q_target.load_state_dict(Q.state_dict())  #Reset Q_target to Q every C steps
                    # Updates #
                    epsilon = epsilon_decay * epsilon
                    state = new_state
                    w = w-1

                    if done:
                        break

                    if episode==NUM_EPISODES-1: # we store the total loss for the last episode
                        LOSSES.append(loss.item())

            mean_loss = np.mean(LOSSES)
            print(f'Training complete for C = {C}, Batch Size = {batch_size} - Gamma = {gamma} - Episode {episode+1}/{NUM_EPISODES} - Mean Loss {mean_loss}')
            PARAM_LOSS.append([mean_loss,C,batch_size,gamma])

print('\n')
print('Grid Search Training complete.')

PARAM_LOSS = np.array(PARAM_LOSS)
C = PARAM_LOSS[np.argmin(PARAM_LOSS[:,0])][1]
batch_size = PARAM_LOSS[np.argmin(PARAM_LOSS[:,0])][2]
gamma = PARAM_LOSS[np.argmin(PARAM_LOSS[:,0])][3]
print('Best parameters are C =',C,'and Batch Size =',batch_size,'and Gamma =',gamma)
# save the best parameters
np.save(f'{TICKER}-best_parameters.npy',np.array([C,batch_size,gamma]))
print('Best parameters saved as',f'{TICKER}-best_parameters.npy')

#seaborn plot of the losses in function of C
df = pd.DataFrame(PARAM_LOSS,columns=['Total Loss','C','Batch Size','Gamma'])
sns.relplot(data=df,x='C',y='Total Loss',kind='line',hue='Batch Size',marker='o',palette='viridis')
plt.show()

sns.relplot(data=df,x='Gamma',y='Total Loss',kind='line',hue='C',marker='o',palette='viridis',legend='full')
plt.show()

#Train Q with the best parameters
print('Training with the best parameters ...')
Q = DQN(input_dim, hidden_dim, output_dim)
Q_target = DQN(input_dim, hidden_dim, output_dim)
Q_target.load_state_dict(Q.state_dict()) # theta_target = theta at time 0 (and every C steps)
optimizer = optim.Adam(Q.parameters())
for episode in range(NUM_EPISODES):
    # Initialize environment
    state = env.reset()
    N.clear()
    state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    w = 0
    for t in range(n,T-tw):
        if w == 0:
            if random.random() < epsilon:
                action = random.choice(ACTION_SET)
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

        # Sample random minibatch from replay memory
        minibatch = random.sample(N, min(len(N), int(batch_size)))  # sample random minibatch from replay memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)  # (s_t, a_t, r_t, s_{t+1}, done)

        # Compute Q values for current states with Q-network
        Q_values = Q(torch.cat(state_batch))  # Compute Q values for current states with Q-network
        next_Q_values = Q_target(torch.cat(next_state_batch))  # Compute Q values for next states with target network
        max_next_Q_values, _ = next_Q_values.max(dim=2) # Compute max Q values for next states with target network (because Bellman needs max!)

        #Compute the target Q values
        targets = np.array(reward_batch) + gamma*np.squeeze(max_next_Q_values.detach().numpy()) # y_i = r_i + gamma*max_a' Q_target(s',a')
        #print('max_next_Q_values',max_next_Q_values,max_next_Q_values.shape) #uncomment to check the values are not constant 
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
        # Updates #
        epsilon = epsilon_decay * epsilon
        state = new_state
        w = w-1

        if done:
            break
        
print('Saving the model ...')
torch.save(Q.state_dict(), f'OptimalParams-{TICKER}-Q.pt')
print('Model saved as',f'OptimalParams-{TICKER}-Q.pt')

print('\n')
print('Launching Out of Sample Testing  ...')
print('\n')
# Out of Sample Testing
data_bis = yf.download(TICKER,OOS_START_DATE, OOS_END_DATE, interval=tf)

# Scaling
data_bis = (data_bis - ohlcv_mean) / ohlcv_std # min max normalization with the same min and max as in sample

# epsilon greedy for action selection in OOS as an example # just an example of how to use the trained model
epsilon_oos = 0.01
for t in range(n,len(data_bis)-tw):
    state = data_bis.iloc[t-n:t]
    state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
    if random.random() < epsilon_oos:
        action = random.choice(ACTION_SET)
    else:
        action = torch.argmax(Q(state)).item()-1
    ACTIONS_OOS.append(action)

# Plot the actions and the Close Price to visualize the strategy
data_bis['Close'].plot()
# if action == 1 (buy), plot a triangle pointing up else a triangle pointing down
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