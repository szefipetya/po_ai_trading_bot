
import json
from operator import imod
from bot import Interface, CustomAgent,CustomEnv
from indicators import AddIndicators
import pandas as pd
from tensorflow.keras.optimizers import Adam
from multiprocessing_env import test_multiprocessing,train_multiprocessing
def getJson(df):
    result = df.to_json(orient='columns')
    parsed = json.loads(result)
    formattedJson = json.dumps(parsed, indent=4) 

    return parsed
class Service:
    def run(self):

        df = pd.read_csv('../data/BTCUSD_1h.csv')
        self.df = df.sort_values('Date')
        self.df= AddIndicators(self.df)
        lookback_window_size = 50
        test_window =720*3
        self.train_df = self.df[100:-test_window-lookback_window_size]
        self.test_df = self.df[-test_window-lookback_window_size:]  # 30 days
        agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, optimizer=Adam, batch_size = 32, model="Dense")
        #train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
        #train_agent(train_env, agent, visualize=False, train_episodes=50000, training_batch_size=500)

       # self.test_env = CustomEnv(self.test_df, lookback_window_size=lookback_window_size, Show_reward=True, Show_indicators=True)
        #Interface.test_agent(self.test_env,self.agent, visualize=False, test_episodes=1, folder="2021_01_18_22_18_Crypto_trader", name="1933.71_Crypto_trader", comment="")
        train_multiprocessing(CustomEnv, agent, self.train_df, num_worker = 2, training_batch_size=500, visualize=False, EPISODES=1000)

        self.dfs, self.env_orders= test_multiprocessing(CustomEnv, agent, self.test_df, num_worker =1, visualize=False, test_episodes=1, folder="2021_01_21_20_06_Crypto_trader", name="1984.93_Crypto_trader", comment="Dense")

    def get_df(self):
        return getJson(self.dfs[0])
    def get_order_info(self):
        return getJson(self.env_orders[0])
        