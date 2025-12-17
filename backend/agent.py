from langchain_ollama import ChatOllama

# Initialize the model
llm = ChatOllama(model="phi3")

# Define tools (currently empty as in app2.py, but structure allows adding them)
# Define tools (currently empty as in app2.py, but structure allows adding them)
tools = []

# Finance Glossary (Extracted for shared use)
finance_glossary = {
    "ebitda": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It measures operating profitability.",
    "liquidity ratio": "Liquidity ratios measure a company's ability to pay short-term financial obligations.",
    "derivatives": "Derivatives are financial contracts whose value is linked to an underlying asset like stocks, bonds, or commodities.",
    "repo rate": "Repo rate is the rate at which a central bank lends short-term funds to commercial banks."
}

def get_agent_response(message: str) -> str:
    key = message.lower().strip()
    if key in finance_glossary:
        return f"📚 Glossary Result: {finance_glossary[key]}"

    # Fallback to LLM
    try:
        # Direct invocation since we have no tools yet, which is faster and cleaner for simple chat
        response = llm.invoke(message)
        return response.content
    except Exception as e:
        if "No connection could be made" in str(e) or "10061" in str(e):
             return "🧠 AI Offline: Please start Ollama on your machine (run `ollama run phi3`)."
        return f"Error communicating with AI: {str(e)}"
