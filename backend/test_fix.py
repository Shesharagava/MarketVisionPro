
import yfinance as yf
import pandas as pd
from prophet import Prophet

def test_fix():
    symbol = "AAPL"
    print("Testing Fix...")
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    
    # FIX: Flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print("Columns after flatten:", df.columns)
    
    df = df.reset_index()
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df = df[["ds", "y"]]
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    
    print("Prepared DF head:", df.head())
    
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    print("Success! Forecast tail:", forecast.tail(2)[["ds", "yhat"]])

test_fix()
