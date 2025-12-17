#FinBERT Sentiment Dashboard using Streamlit

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# App title
st.title("💹 FinBERT Financial Sentiment Dashboard")
st.write("Analyze the **sentiment of financial news, reports, or tweets** using FinBERT (ProsusAI).")

# Input
text = st.text_area("Enter a financial statement or headline:")

if st.button("Analyze Sentiment"):
    with st.spinner("Analyzing..."):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        labels = ["positive", "negative", "neutral"]
        sentiment = labels[torch.argmax(probs)]
        prob_dict = dict(zip(labels, probs[0].tolist()))

        # Display results
        st.subheader("📊 Sentiment Result")
        st.write(f"**Predicted Sentiment:** {sentiment.upper()}")

        st.write("**Probabilities:**")
        st.bar_chart(prob_dict)

st.markdown("---")
st.caption("Powered by FinBERT (ProsusAI) — optimized for financial domain NLP.")


