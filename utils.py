import yfinance as yf

def get_stock_data(ticker):
    df = yf.download(ticker, period="60d", interval="1d")
    df.reset_index(inplace=True)
    return df
