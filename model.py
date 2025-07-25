import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_features(df):
    df["Return"] = df["Close"].pct_change()
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    df = df.dropna()
    return df

def train_model(df):
    X = df[["Open", "High", "Low", "Close", "Volume"]]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc
