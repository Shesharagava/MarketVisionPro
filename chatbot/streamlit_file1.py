import streamlit as st

st.title("Stock Price Predictor")
stock = st.text_input("Enter Stock Symbol:")
if st.button("Predict"):
    st.write(f"Predicted price for {stock} is $120.")
