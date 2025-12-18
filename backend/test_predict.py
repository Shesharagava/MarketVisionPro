
import yfinance as yf
import pandas as pd
from prophet import Prophet
import json

def test_predict():
    symbol = "AAPL"
    try:
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        print("Columns:", df.columns)
        print("Head:", df.head())
        
        if df.empty:
            print("DF Empty")
            return

        df = df.reset_index()
        # Check if 'Close' is in columns or if it is a MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            print("MultiIndex detected")
            # Flatten or access correctly
            try:
                # Attempt to access Close directly if it works, or handle multiindex
                 if "Close" in df.columns:
                     print("Close is in columns (top level?)")
            except:
                pass
        
        # Original code logic
        try:
            df = df.rename(columns={"Date": "ds", "Close": "y"})
            df = df[["ds", "y"]]
            print("Renamed and filtered:", df.head())
        except Exception as e:
            print(f"Rename failed: {e}")
            # If MultiIndex, 'Close' might not be found simply like that if it was under the Ticker
            return

        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["y"])
        
        if len(df) < 30:
            print("Not enough data")
            return

        model = Prophet(daily_seasonality=True)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)
        
        results = forecast.tail(7)[["ds", "yhat"]]
        print("Forecast tail:", results)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_predict()
