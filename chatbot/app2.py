import streamlit as st
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# Initialize the model (make sure you have pulled a model in Ollama, e.g., `ollama pull phi3`)
llm = ChatOllama(model="phi3")

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a finance data assistant."
)

st.set_page_config(page_title="Finance LLM Assistant", layout="centered")
st.title("🪙 Finance Data Assistant (Local LLM)")

# -------------------------------------------
# FinKnowledge Search Section
# -------------------------------------------
st.subheader("📚 FinKnowledge Search (Keyword Based Lookup)")

fin_search = st.text_input("Search Financial Term / Company / Market Concept:", placeholder="Example: EBITDA, Liquidity ratio, Derivatives...")

# Here you can later connect RAG, DB, APIs. For now, we provide a placeholder.
finance_glossary = {
    "ebitda": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization. It measures operating profitability.",
    "liquidity ratio": "Liquidity ratios measure a company's ability to pay short-term financial obligations.",
    "derivatives": "Derivatives are financial contracts whose value is linked to an underlying asset like stocks, bonds, or commodities.",
    "repo rate": "Repo rate is the rate at which a central bank lends short-term funds to commercial banks."
}

if fin_search:
    key = fin_search.lower()
    if key in finance_glossary:
        st.success(finance_glossary[key])
    else:
        st.info("Term not found in local glossary. You can connect external market APIs later.")
