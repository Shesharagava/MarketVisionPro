import streamlit as st
from sentiment import analyze_sentiment

st.set_page_config(page_title="Finance Sentiment Bot", layout="centered")

st.title("🪙 Finance Sentiment Chatbot")
st.write("Analyze *market news, tweets, company updates, trading statements* right here.")

# Chat history store
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input box
user_input = st.chat_input("Type market news or a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Sentiment analysis
    sentiment = analyze_sentiment(user_input)

    bot_response = (
        f"**Sentiment Analysis Result**:\n\n"
        f"- FinBERT: **{sentiment['FinBERT Label']}** "
        f"(Confidence: {sentiment['FinBERT Confidence']})\n"
        f"- VADER Compound Score: **{sentiment['VADER Compound']}**"
    )

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    with st.chat_message("assistant"):
        st.write(bot_response)
