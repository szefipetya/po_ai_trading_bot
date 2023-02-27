
import json
from operator import imod
from os import remove
from bot import Interface, CustomAgent,CustomEnv
from indicators import AddIndicators,indicators_dataframe
import pandas as pd
from tensorflow.keras.optimizers import Adam
from multiprocessing_env import test_multiprocessing,train_multiprocessing, load_agent, continue_train_multiprocessing
from utils import Normalizing
from tensorflow.keras.optimizers import Adam, RMSprop

def getJson(df):
    result = df.to_json(orient='columns')
    parsed = json.loads(result)
    formattedJson = json.dumps(parsed, indent=4) 

    return parsed
class Service:
    def run(self):

        df = pd.read_csv('./BTCUSD_1h.csv')
        df = df.dropna()
        df = df.sort_values('Date')

        df = indicators_dataframe(df, threshold=0.5, plot=False) # insert indicators to df 2021_02_18_21_48_Crypto_trader
        #df = AddIndicators(df) # insert indicators to df 2021_02_21_17_54_Crypto_trader
        print('columns:', df.columns)
        depth = len(list(df.columns[1:])) # OHCL + indicators without Date

        df_nomalized = Normalizing(df[99:])[1:].dropna()
        df = df[100:].dropna()

        lookback_window_size = 100
        test_window =720*3 # 3 months 23000
        
        # split training and testing datasets
        train_df = df[:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
        test_df = df[-test_window-lookback_window_size:]
        
        # split training and testing normalized datasets
        train_df_nomalized = df_nomalized[:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
        test_df_nomalized = df_nomalized[-test_window-lookback_window_size:]

        # single processing training
        #agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, optimizer=Adam, batch_size = 32, model="CNN")
        #train_env = CustomEnv(df=train_df, df_normalized=train_df_nomalized, lookback_window_size=lookback_window_size)
        #train_agent(train_env, agent, visualize=False, train_episodes=50000, training_batch_size=500)

        # multiprocessing training/testing. Note - run from cmd or terminal
       # agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, optimizer=Adam, batch_size=64, model="CNN", depth=depth, comment="Normalized epochs=4, optimizer=RMSprop, batch_size=256, model='CNN', with all indicator")
        #train_multiprocessing(CustomEnv, agent, train_df, train_df_nomalized, num_worker = 14, training_batch_size=500, visualize=False, EPISODES=600000)
        #agent, train_state = load_agent(CustomAgent,"2022_08_28_18_59_Crypto_trader")
       # continue_train_multiprocessing(CustomEnv, agent, train_df, train_df_nomalized, train_state, num_worker = 12, training_batch_size=500, visualize=False, EPISODES=600000,smoothing=10)
       
        #test_multiprocessing(CustomEnv, CustomAgent, test_df, test_df_nomalized, num_worker = 16, visualize=False, test_episodes=1000, folder="2021_02_18_21_48_Crypto_trader", name="3906.52_Crypto_trader", comment="3 months")
       # self.dfs, self.env_orders=test_multiprocessing(CustomEnv, CustomAgent, test_df, test_df_nomalized, num_worker =1, visualize=False, test_episodes=1,batch_size=0, folder="2022_08_28_18_59_Crypto_trader", name="", comment="3 months,ETHUSDT_1h")
        self.dfs, self.env_orders=test_multiprocessing(CustomEnv, CustomAgent, test_df, test_df_nomalized, num_worker =1, visualize=False, test_episodes=1,batch_size=0, folder="2022_08_28_18_59_Crypto_trader", name="", comment="3 months")


    def get_df(self):
        return getJson(self.dfs[0])
    def get_order_info(self):
        return getJson(self.env_orders[0])
