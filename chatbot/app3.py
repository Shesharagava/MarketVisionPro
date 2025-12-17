import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load FinBERT model and tokenizer
@st.cache(allow_output_mutation=True)
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

tokenizer, finbert_model = load_finbert()
vader_analyzer = SentimentIntensityAnalyzer()

def finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ['neutral', 'positive', 'negative']
    sentiment = labels[torch.argmax(probs)]
    confidence = torch.max(probs).item()
    return sentiment, confidence

def vader_sentiment(text):
    scores = vader_analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "positive", compound
    elif compound <= -0.05:
        return "negative", compound
    else:
        return "neutral", compound

# Streamlit UI
st.title("Finance Dashboard with Sentiment Analysis")

st.subheader("📰 News Sentiment Analysis")
news_text = st.text_area("Enter news snippet or headline:")

sentiment_option = st.selectbox("Choose Sentiment Analyzer:", ["FinBERT", "VADER"])

if st.button("Analyze Sentiment") and news_text.strip():
    with st.spinner("Analyzing..."):
        if sentiment_option == "FinBERT":
            sentiment, conf = finbert_sentiment(news_text)
            st.write(f"FinBERT Sentiment: **{sentiment.capitalize()}** (Confidence: {conf:.2f})")
        else:
            sentiment, score = vader_sentiment(news_text)
            st.write(f"VADER Sentiment: **{sentiment.capitalize()}** (Score: {score:.2f})")
elif st.button("Analyze Sentiment"):
    st.warning("Please enter news text.")

st.subheader("📚 Financial Knowledge Search")
fin_search = st.text_input("Search Financial Term:", placeholder="Example: EBITDA")

finance_glossary = {
    # Your glossary terms here
    "ebitda": "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization.",
    "liquidity ratio": "Liquidity ratios measure a company's ability to pay short-term obligations.",
    "derivatives": "Derivatives are financial contracts linked to underlying assets.",
    "repo rate": "Repo rate is the rate at which a central bank lends short-term funds."
}

if fin_search:
    key = fin_search.lower()
    if key in finance_glossary:
        st.success(finance_glossary[key])
    else:
        st.info("Term not found in local glossary. You can connect external APIs later.")
