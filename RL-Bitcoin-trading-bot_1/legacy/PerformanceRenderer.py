from turtle import color, onclick
from unicodedata import name
from matplotlib.figure import Figure
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


def selection_fn(trace,points,selector):
    print("boo")


class PerformanceRenderer:
    def __init__(self, Show_reward=False) -> None:

        pass

    

    def render_performance(self, env, show_indicators):

        fig = self.candlestick_figure(env,show_indicators)
        # trace2 = go.Scatter(
        #     x=env.full_orders_history["Date"], y=env.full_orders_history["Reward"])
        # layout = go.Layout(
        #     xaxis=dict(
        #         showline=True,
        #         showgrid=True,
        #         showticklabels=True,
        #         linecolor='rgb(204, 204, 204)',
        #         linewidth=2,
        #         showspikes = True,
        #         spikemode='across',
        #         spikesnap='cursor',
        #     ),
        #     yaxis=dict(
        #         showline=True,
        #         showgrid=True,
        #         showticklabels=True,
        #         linecolor='rgb(204, 204, 204)',
        #         linewidth=2,
        #     ),
        #     showlegend=True,
        #     hovermode='x'
        # )

        # data = [trace2] 
        # trace2.on_selection(selection_fn)
        # fig2 = dict(data=data, layout=layout)

        fig_secondary = self.secondary_figure(env)
        fw =go.FigureWidget([fig.data[0]])
        scatter = fig.data[0]
        scatter.on_click(selection_fn)
        
        

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
                    figure=fig_secondary,
                )
            ])
        ])
        app.run_server(debug=False)

        #fig.show()
    
    def secondary_figure(self,env):
        fig = make_subplots(rows=3,cols=1, specs=[[{"secondary_y": True,"rowspan":3}], [{}], [{}],])

        fig.add_trace(go.Scatter(
                x=env.full_orders_history['Date'], y=env.df['MACD'], name="macd", line=dict(
                        color='cyan',
                        width=2
                    )), secondary_y=False)
        fig.add_trace(go.Scatter(
                x=env.full_orders_history['Date'], y=env.df['RSI'], name="rsi", line=dict(
                        color='pink',
                        width=2
                    )), secondary_y=False)
        return fig

    def candlestick_figure(self,env, show_indicators=True):
        fig = make_subplots(rows=3,cols=1, specs=[[{"secondary_y": True,"rowspan":3}], [{}], [{}],])
        fig.add_candlestick(x=env.df['Date'],
                            open=env.df['Open'],
                            high=env.df['High'],
                            low=env.df['Low'],
                            close=env.df['Close'])
        # env.full_orders_history=env.full_orders_history.apply(lambda x:)
        fig.add_trace(go.Scatter(
            x=env.full_orders_history['Date'], y=env.full_orders_history['NetWorth'], name="Net worth data", line=dict(
                    color='rgba(0, 0, 255, 0.8)',
                    width=2
                )), secondary_y=True)
        #indicators
        if show_indicators:
            fig.add_trace(go.Scatter(
                x=env.full_orders_history['Date'], y=env.df['sma7'], name="sma7", line=dict(
                        color='rgba(255, 255,255, 0.4)',
                        width=2
                    )), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=env.full_orders_history['Date'], y=env.df['sma25'], name="sma25", line=dict(
                        color='grey',
                        width=2
                    )), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=env.full_orders_history['Date'], y=env.df['sma99'], name="sma99", line=dict(
                        color='black',
                        width=2
                    )), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=env.full_orders_history['Date'], y=env.df['bb_bbm'], name="bb_bbm", line=dict(
                        color='orange',
                        width=1
                    )), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=env.full_orders_history['Date'], y=env.df['bb_bbh'], name="bb_bbh", line=dict(
                        color='orange',
                        width=1
                    )), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=env.full_orders_history['Date'], y=env.df['bb_bbl'], name="bb_bbl", line=dict(
                        color='orange',
                        width=1
                    )), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=env.full_orders_history['Date'], y=env.df['psar'], name="psar", line=dict(
                        color='purple',
                        width=2
                    )), secondary_y=False)
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

       



        