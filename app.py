import streamlit as st
from utils import get_stock_data
from model import create_features, train_model

st.title("📈 Stock Price Movement Predictor")

# Step 1: Ticker input
ticker_input = st.text_input("Enter Stock Ticker (e.g., TCS, ZOMATO, AAPL, TSLA, INFY)").upper()

# Step 2: Initialize df and ticker
df = None
ticker = None

# Step 3: Run only if there's an input and the user presses Predict
if st.button("Predict") and ticker_input:
    with st.spinner("🔍 Fetching data and training model..."):
        # Handle NSE fallback logic
        if "." in ticker_input:
            ticker = ticker_input
        else:
            try:
                df_try = get_stock_data(ticker_input)
                if df_try is None or df_try.empty:
                    ticker = ticker_input + ".NS"
                else:
                    ticker = ticker_input
            except:
                ticker = ticker_input + ".NS"

        # Final fetch with selected ticker
        df = get_stock_data(ticker)

        # Error: No data
        if df is None or df.empty or "Close" not in df.columns:
            st.error("❌ Failed to fetch data. Please check the ticker symbol.")
        else:
            # Display raw data
            st.subheader("📄 Recent Data")
            st.dataframe(df.tail())

            # Step 4: Create features
            df_feat = create_features(df)

            if df_feat.shape[0] < 10:
                st.error("❌ Not enough data to train the model. Try a different ticker or time period.")
            else:
                # Step 5: Train model and predict
                model, acc = train_model(df_feat)

                latest = df_feat[["Open", "High", "Low", "Close", "Volume"]].iloc[-1:]
                prediction = model.predict(latest)[0]

                st.subheader("📊 Prediction for next day:")
                st.markdown(f"**{'📈 UP' if prediction == 1 else '📉 DOWN'}**")

                st.subheader("✅ Model Accuracy")
                st.write(f"{acc * 100:.2f}%")


