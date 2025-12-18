from langchain_ollama import ChatOllama
import yfinance as yf
import re


llm = ChatOllama(model="phi3")


tools = []

finance_glossary = {
    "ebitda": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It measures operating profitability.",
    "liquidity ratio": "Liquidity ratios measure a company's ability to pay short-term financial obligations.",
    "derivatives": "Derivatives are financial contracts whose value is linked to an underlying asset like stocks, bonds, or commodities.",
    "repo rate": "Repo rate is the rate at which a central bank lends short-term funds to commercial banks."
}

def fetch_stock_news(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        news_items = ticker.news
        if not news_items:
            return []
        
        formatted_news = []
        for item in news_items[:5]: # Top 5 news
            # Handle different structure versions of yfinance news
            title = item.get("title")
            publisher = item.get("publisher")
            if not title and "content" in item:
                 # Backup for complex objects
                 title = item["content"].get("title")
            
            if title:
                formatted_news.append(f"- {title} (Source: {publisher})")
        
        return formatted_news
    except Exception as e:
        return [f"Error fetching news: {str(e)}"]

def get_agent_response(message: str) -> str:
    key = message.lower().strip()
    
    # 1. Glossary Check
    if key in finance_glossary:
        return f"📚 Glossary Result: {finance_glossary[key]}"

    # 2. News Intent Check
    context = ""
    if "news" in key:
        # Extract potential ticker: look for uppercase words len 2-5
        # Exclude common words like "THE", "FOR" if they match, but simple regex is fine for now
        # Default to SPY (Market) if no symbol found but 'market' is mentioned
        
        # Extract potential ticker: look for words len 2-6
        # Exclude common words like "THE", "FOR" if they match
        
        # Use case-insensitive search
        potential_symbols = re.findall(r'\b[a-zA-Z]{2,6}\b', message)
        
        target_symbol = "SPY" # Default market
        
        # Simple filter for common non-tickers
        ignore_list = {"what", "show", "tell", "news", "about", "this", "that", "market", "stock", "price", "real", "time"}
        
        found = None
        for s in potential_symbols:
            if s.lower() not in ignore_list:
                found = s.upper()
                break
        
        if found:
            target_symbol = found
        elif "market" in key:
            target_symbol = "SPY"
        else:
             # If just "news" and no symbol, maybe default to SPY or nothing?
             if "news" == key: 
                 target_symbol = "SPY"
             else:
                 target_symbol = None
             
        if target_symbol:
            news_list = fetch_stock_news(target_symbol)
            if news_list:
                context = f"\n\n[Real-time News for {target_symbol}]:\n" + "\n".join(news_list)

    # 3. LLM Generation
    try:
        prompt = message
        if context:
            prompt = f"""You are a helpful financial assistant with access to real-time news.
Use the following news to answer the user's question if relevant.

{context}

User Question: {message}"""
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        if "No connection could be made" in str(e) or "10061" in str(e):
             return "🧠 AI Offline: Please start Ollama on your machine (run `ollama run phi3`)."
        return f"Error communicating with AI: {str(e)}"
