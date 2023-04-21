import numpy as np
import plotly.io as pio
import streamlit as st

st.set_page_config(page_title="Cryptocurrency Predictor",page_icon='https://clipground.com/images/cryptocurrency-icons-clipart.png',)
with st.spinner("Loading Data"):
  coins = ['Bitcoin', 'Bitcoin SV', 'Bitcoin Cash', 'Ethereum', 'Ethereum Classic', 'Cardano', 'Litecoin', 'Dogecoin', 'Bitcoin Gold', 'Ripple', 'Stellar', 'EOS', 'Binance Coin', 'Huobi Token', 'OKB']
  st.title("Model Evaluation Results")
  l, r = st.columns(2)
  accuracy = 88.75928509591448
  rmse = 0.12702707795912385
  l.metric(label="Accuaracy", value=f'{accuracy:.0f}%',delta=f'{accuracy:.05f}')
  r.metric(label="Root Mean Square Error", value=f'{rmse:.03f}',delta=f'-{rmse:.09f}', delta_color='inverse')
  st.divider()
  st.sidebar.title("Model Evaluation Results")
  progress_bar = st.sidebar.progress(0, text="Loading Data")
  for p, c in enumerate(coins, 1):
    progress_bar.progress((p*100)//15, text=f"Loading {c} Data")
    fig = pio.read_json(f"S:\Project\Final_Project\plots\{c.replace(' ', '_')}.json")
    st.subheader(c)
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
  progress_bar.empty()
  # if np.random.choice([True, False], size=1)[0]:
  #     st.balloons()
  # else:
  #     st.snow()