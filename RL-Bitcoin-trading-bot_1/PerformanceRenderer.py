from Env import Env
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sqlalchemy import false, func, null
import vectorbt as vbt

class PerformanceRenderer:
    def render_performance(self, env:Env):
        env.prepare_for_render()
        # prepare data
        portfolio_entries = pd.Series(env.df_entries.set_index("date")["Boolean"].astype('bool'))
        portfolio_exits = pd.Series(env.df_exits.set_index("date")["Boolean"].astype('bool'))
        #portfolio_df = env.df.rename(columns={'Date': 'date','High':'high','Low':'low','Open':'open','Close':'close','Volume':'volume'})
        #portfolio_df['timestamp']=portfolio_df.apply(lambda x: datetime.timestamp(datetime.strptime(x['date'],TIME_FORMAT)),axis=1)
        portfolio_df=env.full_orders_history.rename(columns={'Date':'date','CurrentPrice':'close'})
        portfolio_price = portfolio_df.set_index("date")["close"]
        
        pf = vbt.Portfolio.from_signals(
            portfolio_price, portfolio_entries, portfolio_exits)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_candlestick(x=env.df['Date'],
                            open=env.df['Open'],
                            high=env.df['High'],
                            low=env.df['Low'],
                            close=env.df['Close'])
        #  env.full_orders_history=env.full_orders_history.apply(lambda x:)
        fig.add_trace(go.Scatter(
            x=env.full_orders_history['Date'], y=env.full_orders_history['NetWorth'], name="Net worth data"), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=env.full_orders_history['Date'],
            y=env.full_orders_history['CurrentPrice'], name="Trace",
            mode="markers",
            marker=dict(
                color=env.full_orders_history["Action"].apply(
                    lambda x: "red" if x == 2 else "green"),
                opacity=env.full_orders_history["Action"].apply(
                    lambda x: 0 if x == 0 else 1),
                size=15,
                symbol=env.full_orders_history["Action"].apply(lambda x: x+4)
            )))

        fig.show()
        pf.plot().show()

            # doing this 500 times, each time there are 500 steps made within a random frame