import torch
import torch.nn as nn
import yfinance as yf

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #print("shape of x in forward function DQN :",x.shape)
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class StockTradingEnv:
    def __init__(self, ticker, start, end, lookback,tw):
        self.data = yf.download(ticker, start, end)
        self.lookback = lookback
        self.tw = tw
        self.reset()

    def reset(self):
        self.t = self.lookback  # current time step
        return self.get_state(self.lookback)

    def step(self, action):

        price_now = self.data['Close'].iloc[self.t]
        if self.t + self.tw < len(self.data):
            price_next = self.data['Close'].iloc[self.t + self.tw]
        else:
            price_next = price_now  # Set a default value if out-of-bounds
        reward = action * (price_next - price_now) / price_now

        # Move to next time step
        self.t += 1

        done = False  # Set done to False by default
                
        # Update done flag based on your condition
        if 2 == 1:  ## ADD CONDITION 
            done = True
                
        return self.get_state(self.lookback), reward, done

    def get_state(self, n):
        #return self.data['Close'].iloc[self.t - n:self.t] # uncomment to get only C in OHLCV data #if so please change input_dim in main.py
        return self.data.iloc[self.t - n:self.t]