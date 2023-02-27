function unpack(rows, key) {
    return rows(function(row) {
      return row[key];
    });
  }


const userAction = async () => {
    const df_ = await fetch('http://127.0.0.1:8000/df', {
    });
    const full_orders_history_ = await fetch('http://127.0.0.1:8000/order_info', {
    });

    let full_orders_history = await full_orders_history_.json(); //extract JSON from the http response
    let df = await df_.json(); //extract JSON from the http response
    //resp=JSON.parse(k);
    // do something with myJson
    
    console.log(df)
    console.log(full_orders_history)
    console.log(trace1)
        var trace1 = {
            name: 'yaxis1 data',
            x:Object.values(df['Date']),
            
            close: Object.values(df['Close']),
            high:Object.values(df['High']),
            low: Object.values(df['Low']),
            open:Object.values(df['Open']),
            type: 'candlestick',
            yaxis:'y'
        }

        
          var trace2 = {
            x:Object.values(full_orders_history['Date']),
            y: Object.values(full_orders_history['NetWorth']),
            name: 'yaxis2 data',
            yaxis: 'y2',
            type: 'scatter',
            line:{
                color:'blue',
                width:2.5
            }
          };


          //auxiliary traces
          let keys=["sma7","trend_macd","trend_macd_diff","trend_mass_index","trend_adx","trend_psar_up_indicator","trend_psar_down_indicator","volatility_bbm","volatility_bbw"]
          let colors=["rgba(255, 255,255, 0.4)","grey","black","orange","green","aqua","purple","red","lightblue"]
          let traces=[];

          for (const [key, value] of Object.entries(df)) {
            var randomColor = Math.floor(Math.random()*16777215).toString(16);
            traces.push( {
              x:Object.values(df['Date']),
              y: Object.values(df[key]),
              name: key+' data',
              yaxis: 'y',
              type: 'scatter',
              line:{
                  color:randomColor,
                  width:2.5
              }
            });
          }

          var order_marker = {
            x:Object.values(full_orders_history['Date']),
            y: Object.values(full_orders_history['CurrentPrice']),
            name: 'marker data',
            mode: 'markers+text',
            text:Object.values(full_orders_history["Reward"]).map((x) => {
               if (x<1 &&x>-1) return "";
               else return Math.round(x,1)
            }),
            textposition:'top center',
            marker:{
                color: Object.values(full_orders_history["Action"]).map((x) => {
                    if(x==1) return "green"
                    if(x==2) return "red"
                    if(x==0) return ""
                }),
                opacity: Object.values(full_orders_history["Action"]).map((x) => {
                    if(x==0) return 0;
                    else return 1;
                }),
                size:15,
                symbol: Object.values(full_orders_history["Action"]).map((x) => {
                    return x+4;
                })
                },
            yaxis: 'y',
            type: 'scatter'
          };




        var layout = {
            dragmode: 'pan',
            showlegend: true,
            height:600,
            margin:{
                b:20,
                t:20,
                l:50,
                r:0,
                pad:2
            },
            xaxis: {
              rangeslider: {
                   visible: false
               },
               range:[Object.values(full_orders_history['Date'])[1500],Object.values(full_orders_history['Date'])[2100]]
            },
            yaxis: {
                title: 'yaxis title',
                anchor:'free',
                side:'left'
              },
              yaxis2: {
                title: 'yaxis5 title',
                titlefont: {color: '#9467bd'},
                tickfont: {color: '#9467bd'},
                overlaying: 'y',
                side: 'right',
                anchor: 'free'
              }
          };
        var layout2 = {
            dragmode: 'pan',
            showlegend: true,
            height:350,
            margin:{
                b:20,
                t:20,
                l:50,
                r:0,
                pad:0
            }
        }
        var layout3 = {
          dragmode: 'pan',
          showlegend: true,
          height:350,
          margin:{
              b:20,
              t:20,
              l:50,
              r:0,
              pad:0
          }
      }
        var data = traces.filter(t=>{//around price
          return t.name.startsWith("sma7","")
        }).concat([trace1,trace2,order_marker]);//[trace1,trace2,order_marker,trace_sma7,trace_sma25,trace_sma99,trace_bb_bbh,trace_bb_bbm,trace_bb_bbl,trace_psar];
        var data2 =  traces.filter(t=>{//0-100
          return t.name.startsWith("momentum_rsi")||
          t.name.startsWith("momentum_pvo")||
          t.name.startsWith("volume_mfi")||
          t.name.startsWith("volatility_bbw")
        })
        var data3 =  traces.filter(t=>{//0-1
          return t.name.startsWith("momentum_stoch_rsi")||
           t.name.startsWith("volatility_bbhi")||
           t.name.startsWith("volatility_bbp")||
           t.name.startsWith("trend_psar_down_indicator")||
           t.name.startsWith("trend_psar_up_indicator")
        })
        Plotly.newPlot('myDiv', data, layout,{scrollZoom: true});
        Plotly.newPlot('myDiv2', data2, layout2,{scrollZoom: true});
        Plotly.newPlot('myDiv3', data3, layout3,{scrollZoom: true});

        var myPlot = document.getElementById('myDiv');
        var myPlot2 = document.getElementById('myDiv2');
        var myPlot3 = document.getElementById('myDiv3');

        myPlot.on('plotly_afterplot', function(){
            myPlot2.layout.xaxis.range=myPlot.layout.xaxis.range
            Plotly.restyle(myPlot2, myPlot2.layout)
        });
  
}
userAction() 


