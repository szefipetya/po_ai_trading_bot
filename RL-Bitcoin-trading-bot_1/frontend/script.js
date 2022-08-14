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
          var trace_sma7 = {
            x:Object.values(df['Date']),
            y: Object.values(df['sma7']),
            name: 'sma7 data',
            yaxis: 'y',
            type: 'scatter',
            line:{
                color:'rgba(255, 255,255, 0.4)',
                width:2.5
            }
          };
          var trace_sma25 = {
            x:Object.values(df['Date']),
            y: Object.values(df['sma25']),
            name: 'sma25 data',
            yaxis: 'y',
            type: 'scatter',
            line:{
                color:'grey',
                width:2.5
            }
          };
          var trace_sma99 = {
            x:Object.values(df['Date']),
            y: Object.values(df['sma99']),
            name: 'sma99 data',
            yaxis: 'y',
            type: 'scatter',
            line:{
                color:'black',
                width:2.5
            }
          };
          var trace_bb_bbm = {
            x:Object.values(df['Date']),
            y: Object.values(df['bb_bbm']),
            name: 'bb_bbm data',
            yaxis: 'y',
            type: 'scatter',
            line:{
                color:'orange',
                width:1
            }
          };
          var trace_bb_bbh = {
            x:Object.values(df['Date']),
            y: Object.values(df['bb_bbh']),
            name: 'bb_bbh data',
            yaxis: 'y',
            type: 'scatter',
            line:{
                color:'orange',
                width:1
            }
          };
          var trace_bb_bbl = {
            x:Object.values(df['Date']),
            y: Object.values(df['bb_bbl']),
            name: 'bb_bbl data',
            yaxis: 'y',
            type: 'scatter',
            line:{
                color:'orange',
                width:1
            }
          };
          var trace_psar = {
            x:Object.values(df['Date']),
            y: Object.values(df['psar']),
            name: 'psar data',
            yaxis: 'y',
            type: 'scatter',
            line:{
                color:'purple',
                width:1.5
            }
          };
          var trace_rsi = {
            x:Object.values(df['Date']),
            y: Object.values(df['RSI']),
            name: 'psar data',
            yaxis: 'y',
            type: 'scatter',
            line:{
                color:'grey',
                width:2
            }
          };




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
            height:700,
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
            height:300,
            margin:{
                b:20,
                t:20,
                l:50,
                r:0,
                pad:0
            }
        }
        var data = [trace1,trace2,order_marker,trace_sma7,trace_sma25,trace_sma99,trace_bb_bbh,trace_bb_bbm,trace_bb_bbl,trace_psar];
        var data2 = [trace_rsi]
        Plotly.newPlot('myDiv', data, layout,{scrollZoom: true});
        Plotly.newPlot('myDiv2', data2, layout2,{scrollZoom: true});

        var myPlot = document.getElementById('myDiv');
        var myPlot2 = document.getElementById('myDiv2');

        myPlot.on('plotly_afterplot', function(){
            myPlot2.layout.xaxis.range=myPlot.layout.xaxis.range
            Plotly.restyle(myPlot2, myPlot2.layout)
        });
  
}
userAction() 


