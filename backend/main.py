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

        # Flatten columns if MultiIndex (fix for new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

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
        results["ds"] = results["ds"].dt.strftime('%Y-%m-%d')
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
        
        agent_resp = get_agent_response(request.message)

        final_response = agent_resp

        return {
            "response": final_response,
            "sentiment": sentiment
        }
    except Exception as e:
        return {"error": str(e)}




if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

