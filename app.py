import streamlit as st
from utils import get_stock_data
from model import create_features, train_model

st.title("📈 Stock Price Movement Predictor")

ticker_input = st.text_input("Enter Stock Ticker (e.g., TCS.NS, ZOMATO.NS, AAPL, TSLA, INFY)").upper()
if "." not in ticker_input:
    ticker = ticker_input + ".NS"
else:
    ticker = ticker_input


if st.button("Predict"):
    with st.spinner("Fetching data and training model..."):
        df = get_stock_data(ticker)
        st.subheader("Recent Data")
        st.dataframe(df.tail())

        df_feat = create_features(df)
        model, acc = train_model(df_feat)

        latest = df_feat[["Open", "High", "Low", "Close", "Volume"]].iloc[-1:]
        prediction = model.predict(latest)[0]

        st.subheader("📊 Prediction for next day:")
        st.markdown(f"**{'📈 UP' if prediction == 1 else '📉 DOWN'}**")

        st.subheader("✅ Model Accuracy")
        st.write(f"{acc * 100:.2f}%")
