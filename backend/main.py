from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import uvicorn
from prophet import Prophet
import numpy as np
from pydantic import BaseModel
import random
import datetime

app = FastAPI(title="MarketVision Pro Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def root():
    return {"message": "MarketVision Pro backend is running!"}


@app.get("/api/quote")
def get_realtime_quote(symbol: str = Query(...)):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.history(period="1d", interval="1m")

        if info.empty:
            return {"error": "No recent data available"}

        latest = info.iloc[-1]
        
        prev_data = ticker.history(period="2d")
        if len(prev_data) < 2:
            return {"error": "Insufficient historical data for comparison"}
        
        prev_close = prev_data.iloc[-2]["Close"]
        change = latest["Close"] - prev_close
        change_percent = (change / prev_close) * 100

        return {
            "symbol": symbol,
            "price": round(latest["Close"], 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "ts": int(latest.name.timestamp())
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/daily")
def get_historical_data(symbol: str = Query(...), outputsize: str = "compact"):
    try:
        days = 100 if outputsize == "compact" else 365
        df = yf.download(symbol, period=f"{days}d", interval="1d")

        if df.empty:
            return {"error": "No historical data available"}

        labels = [d.strftime("%Y-%m-%d") for d in df.index]
        closes = [round(v[0], 2) for v in df["Close"].values.tolist()]
        return {"labels": labels, "closes": closes}

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/predict")
def predict_future(symbol: str = Query(...), days: int = Query(7)):
    try:
        ("In Try of api/predict")
        df = yf.download(symbol, period="1y", interval="1d")
        if df.empty:
            return {"error": "No data available", "forecast": []}

        df = df.reset_index()
        df = df.rename(columns={"Date": "ds", "Close": "y"})
        df = df[["ds", "y"]]
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["y"])
        
        if len(df) < 30:
            return {"error": "Insufficient data for prediction (need at least 30 days)", "forecast": []}

        try:
            model = Prophet(daily_seasonality=True)
            model.fit(df)
        except Exception as model_error:
            return {"error": f"Model training failed: {str(model_error)}", "forecast": []}

        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        results = forecast.tail(days)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        results["ds"] = results["ds"].astype(str)
        forecast_points = results.to_dict(orient="records")

        return {"symbol": symbol, "forecast": forecast_points}

    except Exception as e:
        return {"error": str(e), "forecast": []}



from pydantic import BaseModel
from sentiment import analyze_sentiment
from sentiment import analyze_sentiment
from agent import get_agent_response, finance_glossary

@app.get("/api/glossary")
def lookup_term(term: str = Query(...)):
    key = term.lower().strip()
    if key in finance_glossary:
        return {"term": term, "definition": finance_glossary[key]}
    else:
        return {"error": "Term not found"}

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    try:
        sentiment = analyze_sentiment(request.message)
        
        # Get response from the AI agent (which covers glossary + LLM)
        agent_resp = get_agent_response(request.message)

        # Determine response-prefix based on sentiment (optional wrapper)
        # We can append the sentiment analysis to the AI response or just return it in the structured field.
        # The frontend uses 'response' for the main bubble.
        
        # Let's combine them gracefully:
        final_response = agent_resp

        return {
            "response": final_response,
            "sentiment": sentiment
        }
    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------------------------
# Backtesting / Strategy Logic
# -------------------------------------------------------------------

def synthetic_price_series(days=365, start_price=100.0, seed=42):
    """Create a synthetic price series (close prices) for quick backtest/demos."""
    np.random.seed(seed)
    # Generate dates. Note: This simple method doesn't skip weekends/holidays, 
    # but sufficient for synthetic demo.
    dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    # Geometric random walk
    returns = np.random.normal(loc=0.0005, scale=0.02, size=days)
    prices = start_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({"ds": dates, "y": prices})
    df.set_index("ds", inplace=True)
    return df

def sma(series, window):
    return series.rolling(window).mean()

def compute_metrics(trades, portfolio_values):
    # Simple metrics: total return, max drawdown
    if not portfolio_values:
        return {}
    start = portfolio_values[0]
    end = portfolio_values[-1]
    total_return = (end / start) - 1.0
    
    # max drawdown
    vals = np.array(portfolio_values)
    running_max = np.maximum.accumulate(vals)
    # Avoid division by zero
    running_max[running_max == 0] = 1 
    drawdowns = (vals - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))
    
    return {
        "total_return": float(round(total_return * 100, 2)), 
        "max_drawdown": float(round(max_drawdown * 100, 2)), 
        "trades_count": len(trades)
    }

class BacktestRequest(BaseModel):
    symbol: str = "SYNTHETIC"
    sma_short: int = 10
    sma_long: int = 50
    capital: float = 10000.0
    days: int = 365

@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    try:
        # 1. Get Data
        if req.symbol.upper() == "SYNTHETIC":
            df = synthetic_price_series(days=req.days, start_price=100.0)
            # rename for consistency
            prices = df["y"]
        else:
            # Fetch real data for larger period to ensure enough for lag
            # Note: yfinance auto-adjusts for weekends, so len might be < days
            ticker = yf.Ticker(req.symbol)
            hist = ticker.history(period="2y", interval="1d")
            if hist.empty:
                return {"error": "No data found for symbol"}
            
            # We take the last N days requested, but need extra for SMA calculation
            hist = hist.tail(req.days + req.sma_long + 10)
            prices = hist["Close"]
            # Convert index to date only if it's datetime
            prices.index = [d.date() for d in prices.index]

        # 2. Compute Indicators
        short_sma = sma(prices, req.sma_short)
        long_sma = sma(prices, req.sma_long)

        # 3. Simulate Strategy
        position = 0  # 0 = cash, 1 = invested
        cash = req.capital
        shares = 0
        portfolio_values = []
        equity_curve = [] # list of {time, value}
        trades = []

        # We iterate through the series. 
        # We need to align everything. 
        # Let's drop NaN from SMAs first to start trading when valid
        valid_indices = short_sma.dropna().index.intersection(long_sma.dropna().index)
        
        # We only trade on valid days
        for date in valid_indices:
            price = float(prices.loc[date])
            s = short_sma.loc[date]
            l = long_sma.loc[date]
            
            # Signal Logic
            # BUY if Short > Long and we have no position
            if s > l and position == 0:
                shares = cash / price
                cash = 0
                position = 1
                trades.append({
                    "date": str(date), 
                    "type": "BUY", 
                    "price": round(price, 2), 
                    "shares": round(shares, 4)
                })
            
            # SELL if Short < Long and we have position
            elif s < l and position == 1:
                cash = shares * price
                shares = 0
                position = 0
                trades.append({
                    "date": str(date), 
                    "type": "SELL", 
                    "price": round(price, 2), 
                    "value": round(cash, 2)
                })
            
            # Update Portfolio Value
            current_val = cash + (shares * price)
            portfolio_values.append(current_val)
            equity_curve.append({"time": str(date), "value": round(current_val, 2)})

        # 4. Finalize
        # Force sell at end to get final cash value if holding? 
        # Or just value it.
        metrics = compute_metrics(trades, portfolio_values)

        return {
            "symbol": req.symbol,
            "metrics": metrics,
            "trades": trades[-50:], # limit size
            "equity_curve": equity_curve
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

