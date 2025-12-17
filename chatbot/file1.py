# single_app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import io
import json
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# ----------------------------
# Utilities / small helpers
# ----------------------------
def synthetic_price_series(days=365, start_price=100.0, seed=42):
    """Create a synthetic price series (close prices) for quick backtest/demos."""
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    # Geometric random walk
    returns = np.random.normal(loc=0.0005, scale=0.02, size=days)
    prices = start_price * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"date": dates, "close": prices})
    df.set_index("date", inplace=True)
    return df

def sma(series, window):
    return series.rolling(window).mean()

def compute_metrics(trades, portfolio_values):
    # Simple metrics: total return, max drawdown
    if len(portfolio_values) == 0:
        return {}
    start = portfolio_values[0]
    end = portfolio_values[-1]
    total_return = (end / start) - 1.0
    # max drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))
    return {"total_return": float(total_return), "max_drawdown": float(max_drawdown), "trades": trades}

# ----------------------------
# Sentiment endpoint (very simple)
# ----------------------------
# This is intentionally tiny: rule-based keyword scoring.
POSITIVE_KEYWORDS = {"gain","positive","up","bull","beats","beat","surge","rally","upgrade"}
NEGATIVE_KEYWORDS = {"down","drop","miss","misses","decline","sell","bear","downgrade","crash"}

@app.route("/sentiment", methods=["POST"])
def sentiment():
    """
    POST JSON: { "text": "news headline or article text" }
    Returns: {"score": -1..1, "label": "positive/neutral/negative", "counts": {...}}
    """
    data = request.get_json(force=True)
    text = data.get("text", "") if data else ""
    tokens = set(text.lower().split())
    pos = len(tokens & POSITIVE_KEYWORDS)
    neg = len(tokens & NEGATIVE_KEYWORDS)
    raw_score = pos - neg
    # normalize: simple scaling
    if raw_score > 0:
        score = min(1.0, raw_score / 3.0)
    elif raw_score < 0:
        score = max(-1.0, raw_score / 3.0)
    else:
        score = 0.0
    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"
    return jsonify({"score": score, "label": label, "pos_count": pos, "neg_count": neg})

# ----------------------------
# Backtest endpoint (simple SMA crossover)
# ----------------------------
@app.route("/backtest", methods=["POST"])
def backtest():
    """
    POST JSON or form-data:
      - Option A: send CSV file (as raw body or form file) with columns date,close
      - Option B: send JSON {"mode":"synthetic","days":365,"start_price":100}
      - Options for strategy: sma_short (int), sma_long (int)
    Returns: basic metrics and an example equity curve
    """
    payload = request.get_json(silent=True) or {}
    # strategy params
    sma_short = int(payload.get("sma_short", 10))
    sma_long = int(payload.get("sma_long", 50))
    mode = payload.get("mode", "synthetic")

    # load data
    if mode == "csv" and "csv" in payload:
        # payload['csv'] is a CSV text
        csv_text = payload["csv"]
        df = pd.read_csv(io.StringIO(csv_text), parse_dates=["date"])
        df.set_index("date", inplace=True)
        if "close" not in df.columns:
            return jsonify({"error": "CSV must contain 'close' column."}), 400
    elif mode == "synthetic":
        days = int(payload.get("days", 365))
        df = synthetic_price_series(days=days, start_price=float(payload.get("start_price", 100.0)))
    else:
        return jsonify({"error": "unknown mode"}), 400

    prices = df["close"].copy()
    # compute SMAs
    short_sma = sma(prices, sma_short)
    long_sma = sma(prices, sma_long)

    position = 0  # 0 = no pos, 1 = long
    entry_price = 0.0
    cash = 100000.0
    shares = 0
    portfolio_values = []
    trades = []

    for date, price in prices.iteritems():
        s = short_sma.loc[date]
        l = long_sma.loc[date]
        # require both SMA exist
        if np.isnan(s) or np.isnan(l):
            # record portfolio value
            pv = cash + shares * price
            portfolio_values.append(pv)
            continue
        # buy signal
        if s > l and position == 0:
            # buy as many shares as fixed fraction (here we buy full position for demo)
            shares = cash // price
            cash -= shares * price
            position = 1
            entry_price = price
            trades.append({"date": str(date.date()), "type": "BUY", "price": float(price), "shares": int(shares)})
        # sell signal
        elif s < l and position == 1:
            cash += shares * price
            trades.append({"date": str(date.date()), "type": "SELL", "price": float(price), "shares": int(shares)})
            shares = 0
            position = 0
        pv = cash + shares * price
        portfolio_values.append(pv)

    metrics = compute_metrics(trades, portfolio_values)
    # include small equity series (sample to limit size)
    equity_series = [{"date": str(d.date()), "equity": float(v)} for d, v in zip(prices.index, portfolio_values)[-100:]]
    response = {"metrics": metrics, "trades": trades, "equity_sample": equity_series}
    return jsonify(response)

# ----------------------------
# Prediction endpoint (dummy example)
# ----------------------------
@app.route("/predict", methods=["GET"])
def predict():
    """
    GET /predict?ticker=XXX&days=1
    Returns a simple naive forecast (last_price * (1 + small random drift))
    Replace this with an LSTM/Transformer inference call in real system.
    """
    ticker = request.args.get("ticker", "SYN")
    days = int(request.args.get("days", 1))
    # In real system: fetch recent data and run model.predict
    last_price = float(request.args.get("last_price", 100.0))
    preds = []
    price = last_price
    for i in range(days):
        # naive random walk forecast for demo
        drift = random.normalvariate(0.0005, 0.01)
        price = price * (1 + drift)
        preds.append({"day": i + 1, "predicted_price": round(price, 4)})
    return jsonify({"ticker": ticker, "predictions": preds})

# ----------------------------
# Health / root
# ----------------------------
@app.route("/")
def root():
    return jsonify({"status": "ok", "time": str(datetime.utcnow())})

# ----------------------------
# Run (development)
# ----------------------------
if __name__ == "__main__":
    # dev server
    app.run(debug=True, port=5000)
