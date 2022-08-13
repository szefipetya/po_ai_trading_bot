
import json
from operator import imod
from bot import Interface, CustomAgent,CustomEnv
from indicators import AddIndicators
import pandas as pd
from tensorflow.keras.optimizers import Adam

def getJson(df):
    result = df.to_json(orient='columns')
    parsed = json.loads(result)
    formattedJson = json.dumps(parsed, indent=4) 

    return parsed
class Service:
    def run(self):

        self.df = pd.read_csv('./pricedata.csv')
        self.df = self.df.sort_values('Date')
        self.df= AddIndicators(self.df)
        lookback_window_size = 50
        test_window =720
        self.train_df = self.df[:-test_window-lookback_window_size]
        self.test_df = self.df[-test_window-lookback_window_size:]  # 30 days
        self.agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, optimizer=Adam, batch_size = 32, model="Dense")
        #train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
        #train_agent(train_env, agent, visualize=False, train_episodes=50000, training_batch_size=500)

        self.test_env = CustomEnv(self.test_df, lookback_window_size=lookback_window_size, Show_reward=True, Show_indicators=True)
        Interface.test_agent(self.test_env,self.agent, visualize=False, test_episodes=1, folder="2021_01_18_22_18_Crypto_trader", name="1933.71_Crypto_trader", comment="")
    def get_df(self,indicators=True):
        return getJson(self.test_env.df)
    def get_order_info(self):
        return getJson(self.test_env.full_orders_history)
        