#================================================================
#
#   File name   : RL-Bitcoin-trading-bot_1.py
#   Author      : PyLessons
#   Created date: 2020-12-02
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : Introduction to trading Crypto with Reinforcement Learning
#
#================================================================
import pandas as pd
import numpy as np
import random
from collections import deque

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sqlalchemy import false, func, null

class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df)-1 
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        #they are data, that are related to the account, to the currency pair
        self.lookback_orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.lookback_market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 10)

        self.full_market_history=[]
        self.full_orders_history=pd.DataFrame(columns=['Date','NetWorth', 'CryptoBought','CryptoSold','CryptoHeld','CurrentPrice','Action'])
        self.full_net_worth_history=[]

        

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        if env_steps_size > 0: # used for training dataset #random frames are selected in the dataframe with size of env_steps_size
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):#fill out the lookback window with data 
            current_step = self.current_step - i
            self.lookback_orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.lookback_market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume']
                                        ])

        state = np.concatenate((self.lookback_market_history, self.lookback_orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.lookback_market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume']
                                    ])
        state = np.concatenate((self.lookback_market_history, self.lookback_orders_history), axis=1)
        return state

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close'])

        if action==1 and self.balance<=0.00000000001:
            action=0
        if action==2 and self.crypto_held<=0:
            action=0
        
        if action == 0: # Hold
            pass

        elif action == 1:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought

        elif action == 2 :
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold


        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.lookback_orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        self.full_orders_history=self.full_orders_history.append({'Date':self.df.loc[self.current_step, 'Date'],
        'NetWorth':self.net_worth,
        'CryptoBought':self.crypto_bought,
        'CryptoSold':self.crypto_sold,
        'CryptoHeld':self.crypto_held,
        'CurrentPrice':current_price,
        'Action':action        
        },
        ignore_index=True)            
            
            
          
        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        state = self._next_observation()
        
        return state, reward, done

    # render environment
    def render(self):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')

    def render_performance(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_candlestick( x=self.df['Date'],
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'])
      #  self.full_orders_history=self.full_orders_history.apply(lambda x:)
        fig.add_trace(go.Scatter(x=self.full_orders_history['Date'], y=self.full_orders_history['NetWorth'], name="Net worth data"),secondary_y=True)
        fig.add_trace(go.Scatter(
        x=self.full_orders_history['Date'],
        y=self.full_orders_history['CurrentPrice'], name="Trace",
        mode="markers",
        marker=dict(
            color=self.full_orders_history["Action"].apply(lambda x: "red" if x==2 else "green"),
            opacity=self.full_orders_history["Action"].apply(lambda x: 0 if x==0 else 1),
            size=15,
            symbol=self.full_orders_history["Action"].apply(lambda x:x+4)
        )))
        
        fig.show()


        #doing this 500 times, each time there are 500 steps made within a random frame
def Random_games(env, train_episodes = 50, train_mode=False,training_batch_size=500):
    average_net_worth = 0

    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        while True:
            env.render()

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)


def Single_game(env):
    state = env.reset()
    average_net_worth = 0

    while True:
            env.render()

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break
    env.render_performance()
    print("average_net_worth:", average_net_worth)

df = pd.read_csv('./pricedata.csv')
df = df.sort_values('Date')
df=df[-4000:]

lookback_window_size = 50
train_df = df[:-1000-lookback_window_size]
test_df = df[-1000-lookback_window_size:] # 30 days

train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size)

#Random_games(train_env, train_episodes = 10,train_mode=True, training_batch_size=500)
#Random_games(test_env, train_episodes = 10,train_mode=False)
Single_game(test_env)
