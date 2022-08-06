from matplotlib.figure import Figure
from Env import Env
from chart_studio import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sqlalchemy import false, func, null
import vectorbt as vbt
import chart_studio.dashboard_objs as dashboard
from IPython.display import Image
import plotly.graph_objs as go
import chart_studio.plotly as py
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import numpy as np


class PerformanceRenderer:
    def render_performance(self, env:Env):
        env.prepare_for_render()

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

        app = Dash(__name__)
        app.layout = html.Div(children=[
            html.H1(children='Hello Dash'),

            html.Div(children='''
                Dash: A web application framework for your data.
            '''),

            dcc.Graph(
                id='example-graph',
                figure=fig
            ),
            dcc.Graph(
                id='g2',
                figure=fig
            )
        ])
        app.run_server(debug=True)

        #fig.show()
    def candlestick_figure(self,env) -> Figure:
        fig = make_subplots(rows=3,cols=1, specs=[[{"secondary_y": True,"rowspan":3}], [{}], [{}]])
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
                size=12,
                symbol=env.full_orders_history["Action"].apply(lambda x: x+4)
            )))
        fig['layout'].update(height=1000, width=1900, title='Candlestick')

        return fig

    def render_performance_vbt(self,env):
        env.prepare_for_render()

        




        portfolio_entries = pd.Series(env.df_entries.set_index("date")["Boolean"].astype('bool'))
        portfolio_exits = pd.Series(env.df_exits.set_index("date")["Boolean"].astype('bool'))
        #portfolio_df = env.df.rename(columns={'Date': 'date','High':'high','Low':'low','Open':'open','Close':'close','Volume':'volume'})
        #portfolio_df['timestamp']=portfolio_df.apply(lambda x: datetime.timestamp(datetime.strptime(x['date'],TIME_FORMAT)),axis=1)
        portfolio_df=env.full_orders_history.rename(columns={'Date':'date','CurrentPrice':'close'})
        portfolio_price = portfolio_df.set_index("date")["close"]
        
        pf = vbt.Portfolio.from_signals(
            portfolio_price, portfolio_entries, portfolio_exits)

        app = Dash(__name__)
        app.layout = html.Div(children=[
            html.H1(children='Hello Dash'),

            html.Div(children='''
                Dash: A web application framework for your data.
            '''),
            dcc.Graph(
                id='candle-graph',
                figure=self.candlestick_figure(env)
            ),
            dcc.Graph(
                id='example-graph',
                figure=pf.trades.plot_pnl()
            ),
            dcc.Graph(
                id='g2',
                figure=pf.plot_value()
            )
        ])
        app.run_server(debug=True)
        #pf.plot().show()

       



        