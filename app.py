import streamlit as st
from utils import get_stock_data
from model import create_features, train_model

st.title("📈 Stock Price Movement Predictor")

ticker_input = st.text_input("Enter Stock Ticker (e.g., TCS, ZOMATO, AAPL, TSLA, INFY)").upper()
if "." in ticker_input:
    ticker = ticker_input
else:
    # Try plain first (for US stocks), then fallback to NSE
    try:
        df = get_stock_data(ticker_input)
        if df is None or df.empty:
            ticker = ticker_input + ".NS"
        else:
            ticker = ticker_input
    except:
        ticker = ticker_input + ".NS"

if st.button("Predict"):
    with st.spinner("Fetching data and training model..."):
        df = get_stock_data(ticker)

if df is None or df.empty or "Close" not in df.columns:
    st.error("❌ Failed to fetch data. Please check the ticker symbol.")
else:
    st.subheader("Recent Data")
    st.dataframe(df.tail())

    df_feat = create_features(df)

    if df_feat.shape[0] < 10:  # need enough rows for train/test split
        st.error("❌ Not enough data to train the model. Try a different ticker or time period.")
    else:
        model, acc = train_model(df_feat)

        latest = df_feat[["Open", "High", "Low", "Close", "Volume"]].iloc[-1:]
        prediction = model.predict(latest)[0]

        st.subheader("📊 Prediction for next day:")
        st.markdown(f"**{'📈 UP' if prediction == 1 else '📉 DOWN'}**")

        st.subheader("✅ Model Accuracy")
        st.write(f"{acc * 100:.2f}%")

