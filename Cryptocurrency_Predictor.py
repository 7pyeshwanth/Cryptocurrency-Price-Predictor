import streamlit as st

st.set_page_config(page_title="Cryptocurrency Predictor",
                   page_icon='https://clipground.com/images/cryptocurrency-icons-clipart.png',)

with st.spinner("Importing library"):
  from yfinance import download as get
  import numpy as np
  import pandas as pd
  from sklearn.preprocessing import MinMaxScaler
  from keras.models import load_model
  from sklearn.metrics import r2_score, mean_squared_error
  import plotly.graph_objs as go
  import datetime

def processData(data, time_period):
  with st.spinner('Prossesing data'):
    x_data, y_data = [], []
    for i in range(time_period, data.shape[0]):
      x_data.append(data[i - time_period:i, 0])
      y_data.append(data[i, 0])
    return np.array(x_data), np.array(y_data)


def predict(coin, no_days):
  with st.spinner('Fetching data'):
    raw_data = get(f"{coins[coin][0]}-USD")
    raw_data.index = pd.to_datetime(raw_data.index)
    raw_data.index = raw_data.index.date
    st.subheader(f'{coin} Description')
    st.table(raw_data.describe())
    st.divider()
    data = pd.DataFrame(raw_data["Adj Close"])
  with st.spinner('Preducting future data'):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train, y_train = processData(scaler.fit_transform(data), 30)
    cytrain = np.array([y_train[-31:-1]])
    f_predict = np.array([])
    for _ in range(no_days):
      f_predict = np.append(f_predict, model[coins[coin][1]-1].predict([cytrain]))
      cytrain = np.array([np.append(cytrain[0][1:], f_predict[-1])])
    date_ind = pd.date_range(
      start=datetime.datetime.now().date(), periods=no_days)
    fpredict = pd.DataFrame(scaler.inverse_transform(
      f_predict.reshape(-1, 1)), index=date_ind, columns=['Predict'],)
  with st.spinner('Ploting future data'):
    fpredict.index = fpredict.index.date
    predicted_graph = go.Figure()
    predicted_graph.add_trace(go.Scatter(
        name="Previous Data",
        x=data.index,
        y=data['Adj Close'],
        line_color='green'
    ))
    predicted_graph.add_trace(go.Scatter(
      name="Predicted Data",
      x=fpredict.index,
      y=fpredict['Predict'],
      mode='lines',
      line_color='red'
    ))
    predicted_graph.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        xaxis_range=[
          data.index[-30], fpredict.index[-1]],
        legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0),
        height=550,
        width=700,
        yaxis=dict(tickformat=',d')
    )
    predicted_graph.update_xaxes( rangeselector=dict(
        buttons=list([
            dict(count=1, label="MONTH", step="month", stepmode="backward", ),
            dict(count=6, label="6 MONTH", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="YEAR", step="year", stepmode="backward"),
            dict(label='ALL', step="all")
        ]), y=1.1,
    ))
    st.subheader(f'{coin} Preduction Graph')
    st.plotly_chart(predicted_graph)
    st.divider()
    st.subheader(f'{coin} Preduction Table')
    st.table(fpredict)
    st.divider()
  
  with st.spinner('Preducting Previous data'):
    y_predict = model[coins[coin][1]-1].predict(x_train)
    ppredict = pd.concat([data, pd.DataFrame(scaler.inverse_transform(
      y_predict), index=data.index[30:], columns=['PPredict'])], axis=1)
  with st.spinner('Ploting Previous data'):
    previous_graph = go.Figure()
    previous_graph.add_trace(go.Scatter(
        name="Actual Data",
        x=ppredict.index,
        y=ppredict['Adj Close'],
        line_color='green'
    ))
    previous_graph.add_trace(go.Scatter(
      name="Predicted Data",
      x=ppredict.index,
      y=ppredict['PPredict'],
      line_color='red'
    ))
    previous_graph.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        xaxis_range=[
          ppredict.index[int(np.floor(len(ppredict['Adj Close']) * 0.5))], ppredict.index[-1]],
        legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="left", x=0),
        height=550,
        width=700,
        yaxis=dict(tickformat=',d')
    )
    previous_graph.update_xaxes(rangeselector=dict(
        buttons=list([
            dict(count=1, label="MONTH", step="month", stepmode="backward", ),
            dict(count=6, label="6 MONTH", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="YEAR", step="year", stepmode="backward"),
            dict(label='ALL', step="all")
        ]), y=1.1,
    ))
    st.subheader(f'{coin} History Price VS Preduction Graph')
    st.plotly_chart(previous_graph,use_container_width=True)
    st.divider()
  with st.spinner('Calulating Accuaracy'):
    rmse = np.sqrt(mean_squared_error(y_train, y_predict))
    accuracy = r2_score(y_train, y_predict) * 100
    l, r = st.columns(2)
    l.metric(label="Accuaracy", value=f'{accuracy:.0f}%',delta=f'{accuracy:.05f}')
    r.metric(label="Root Mean Square Error", value=f'{rmse:.03f}',delta=f'-{rmse:.09f}', delta_color='inverse')

if __name__ == "__main__":
  coins = {'Bitcoin': ('BTC', 1), 'Bitcoin SV': ('BSV', 1), 'Bitcoin Cash': ('BCH', 1), 'Ethereum': ('ETH', 2), 'Ethereum Classic': ('ETC', 2), 'Cardano': ('ADA', 2), 'Litecoin': (
    'LTC', 3), 'Dogecoin': ('DOGE', 3), 'Bitcoin Gold': ('BTG', 3), 'Ripple': ('XRP', 4), 'Stellar': ('XLM', 4), 'EOS': ('EOS', 4), 'Binance Coin': ('BNB', 5), 'Huobi Token': ('HT', 5), 'OKB': ('OKB', 5)}
  model = [load_model(f'models/model{x}.h5') for x in range(1, 6)]
  st.sidebar.title("Cryptocurrency Predictor")
  # st.sidebar.
  st.title("Cryptocurrency Price Predictor")

  coin = st.selectbox(label="Which CryptoCurrency you want to Predict?", options=coins.keys())
  no_days = st.slider(label='Select no of Days?', min_value=1, max_value=100, value=10)
  left, mid, right = st.columns(3)
  left.caption(f'**Coin:  _:green[{coin}]_**')
  diff_date = (datetime.datetime.now().date() + datetime.timedelta(days=10)).strftime("%Y-%m-%d")
  mid.caption(f"**Till Date: _:green[{diff_date:02}]_**")
  right.caption(f'**No of Days: _:green[{no_days}]_**')
  st.divider()

  data = predict(coin, no_days)

  if np.random.choice([True, False], size=1)[0]:
    st.balloons()
  else:
    st.snow()
  st.divider()
