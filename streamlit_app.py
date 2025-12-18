
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import sys
import os

# Ensure backend modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from agent import get_agent_response, finance_glossary
from sentiment import analyze_sentiment

# Set Page Config
st.set_page_config(page_title="MarketVision Pro", page_icon="📈", layout="wide")

# Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Application Logic
def main():
    st.title("📈 MarketVision Pro")
    st.markdown("Live watchlist + Prophet predictions + AI Agent")

    # Sidebar: Watchlist
    st.sidebar.header("Watchlist")
    default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "JPM", "META", "INTC", "KO"]
    
    selected_ticker = st.sidebar.selectbox("Select Ticker", default_tickers)
    days_forecast = st.sidebar.slider("Forecast Days", 1, 30, 7)

    # --- Real-time Data ---
    if selected_ticker:
        try:
            ticker = yf.Ticker(selected_ticker)
            # Fetch 2d to calculate change
            hist = ticker.history(period="2d", interval="1m")
            
            if not hist.empty and len(hist) >= 1:
                latest = hist.iloc[-1]
                price = latest["Close"]
                
                # Check for previous close
                change = 0.0
                pct = 0.0
                if len(hist) >= 2:
                    prev = hist.iloc[-2]["Close"]
                    change = price - prev
                    pct = (change / prev) * 100
                
                # Display Metrics
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Price", f"${price:,.2f}", f"{change:+.2f} ({pct:+.2f}%)")
                with c2:
                    st.metric("High", f"${latest['High']:,.2f}")
                with c3:
                    st.metric("Low", f"${latest['Low']:,.2f}")
            else:
                st.warning("No recent data found for this icon.")
                
            # History + Forecast
            st.divider()
            st.subheader(f"History & Forecast ({days_forecast} days)")
            
            with st.spinner("Fetching data & Training Prophet model..."):
                df = yf.download(selected_ticker, period="1y", interval="1d")
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                df = df.reset_index()
                df = df.rename(columns={"Date": "ds", "Close": "y"})
                df["y"] = pd.to_numeric(df["y"], errors="coerce")
                df = df.dropna(subset=["y"])

                if len(df) > 30:
                    m = Prophet(daily_seasonality=True)
                    m.fit(df)
                    future = m.make_future_dataframe(periods=days_forecast)
                    forecast = m.predict(future)
                    
                    # Plotly Chart
                    fig = go.Figure()
                    
                    # History
                    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='History', line__color='#6366f1'))
                    
                    # Forecast
                    future_df = forecast.tail(days_forecast)
                    fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat'], name='Forecast', line=dict(color='#22d3ee', dash='dash')))
                    fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat_upper'], name='Upper Band', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=future_df['ds'], y=future_df['yhat_lower'], name='Lower Band', line=dict(width=0), fill='tonexty', fillcolor='rgba(34, 211, 238, 0.2)', showlegend=False))

                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        xaxis_gridcolor='rgba(255,255,255,0.1)',
                        yaxis_gridcolor='rgba(255,255,255,0.1)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Not enough data for prediction")

        except Exception as e:
            st.error(f"Error: {e}")

    # --- Chatbot & Glossary ---
    st.divider()
    col_chat, col_glossary = st.columns([2, 1])

    with col_glossary:
        st.subheader("📚 Financial Glossary")
        term = st.text_input("Lookup term (e.g., EBITDA)")
        if st.button("Search Glossary"):
            res = finance_glossary.get(term.lower().strip())
            if res:
                st.success(res)
            else:
                st.error("Term not found.")

    with col_chat:
        st.subheader("🤖 Market Bot (with News)")

        # Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        if prompt := st.chat_input("Ask about market news..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get response from agent
                    response_text = get_agent_response(prompt)
                    
                    # Get sentiment
                    try:
                        sentiment = analyze_sentiment(prompt)
                        # Format sentiment if needed, or just append
                        sentiment_label = sentiment.get("FinBERT Label", "Neutral")
                        st.markdown(response_text)
                        
                        # Add sentiment text in small font
                        st.caption(f"Sentiment Analysis: {sentiment_label}")
                        
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    except:
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
