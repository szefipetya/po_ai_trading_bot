from unicodedata import name
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

from threading import Thread

class PerformanceRenderer:
    def __init__(self, Show_reward=False) -> None:

        pass

    def render_performance(self, env:Env):
        env.prepare_for_render()

        fig = self.candlestick_figure(env)
        trace2 = go.Scatter(
            x=env.full_orders_history["Date"], y=env.full_orders_history["Reward"])
        layout = go.Layout(
            xaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                showspikes = True,
                spikemode='across',
                spikesnap='cursor',
            ),
            yaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
            ),
            showlegend=True,
            hovermode='x'
        )

        data = [trace2]
        fig2 = dict(data=data, layout=layout)
        


        app = Dash(__name__)
        app.layout = html.Div( children=[
            html.H1(children='Hello Dash'),
         html.Div([
                dcc.Graph(
                    id='example-graph',
                    figure=fig,
                )
            ]),
            
            html.Div([
                dcc.Graph(
                    id='example-graph2',
                    figure=fig2,
                )
            ]),
            
        ])
        app.run_server(debug=False)

        #fig.show()
    def candlestick_figure(self,env) -> Figure:
        fig = make_subplots(rows=3,cols=1, specs=[[{"secondary_y": True,"rowspan":3}], [{}], [{}],])
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
            mode="markers+text",
            text=env.full_orders_history['Reward'].apply(lambda x: "" if x<2 and x>-2 else round(x,1) ),
            textposition='top center',
            visible=True,
            marker=dict(
                color=env.full_orders_history["Action"].apply(
                    lambda x: "red" if x == 2 else "green"),
                opacity=env.full_orders_history["Action"].apply(
                    lambda x: 0 if x == 0 else 1),
                size=12,
                symbol=env.full_orders_history["Action"].apply(lambda x: x+4)
            )))
        fig['layout'].update(height=1000, title='Candlestick')
        #fig.update(textpo='top center')

        return fig

    def render_performance_vbt(self,env):
        env.prepare_for_render()

        portfolio_entries = pd.Series(env.df_entries.set_index("date")["Boolean"].astype('bool'))
        portfolio_exits = pd.Series(env.df_exits.set_index("date")["Boolean"].astype('bool'))
        #portfolio_df = env.df.rename(columns={'Date': 'date','High':'high','Low':'low','Open':'open','Close':'close','Volume':'volume'})
        #portfolio_df['timestamp']=portfolio_df.apply(lambda x: datetime.timestamp(datetime.strptime(x['date'],TIME_FORMAT)),axis=1)
        portfolio_df=env.full_orders_history.rename(columns={'Date':'date','CurrentPrice':'close'})
        portfolio_price = portfolio_df.set_index("date")["close"]
        #0.1=10%
        #0.01=1%
        #0.001=0,1%
        #0.0001=0,01%
        
        pf = vbt.Portfolio.from_signals(
            portfolio_price, portfolio_entries, portfolio_exits, fees=np.array([0.0005]),init_cash=1000)
        print(pf)
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
        app.run_server(debug=False)
        
        
        
        #pf.plot().show()

       



        