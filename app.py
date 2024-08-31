import streamlit as st
from backend import get_headlines, apply_probabilities, get_final_class, pred

st.title("S&P 500 Predictor Using NLP")
st.image("header.png", use_column_width=True)
st.write("")
st.markdown("""
        This web application forecasts the movement of the S&P 500 index by the close of the next trading day 
        based on recent financial news headlines from Yahoo Finance. It uses a fine-tuned transformer model to predict 
        whether the index will:
        - increase more than 0.5%
        - decrease more than 0.5%
        - maintain its current level
        
        For more details, visit the [GitHub repository](https://github.com/jeroenvdmbrugge/SP500-Predictor-WebApp).
        """)

if st.button("Click Here to Get Started"):
    df = get_headlines()
    st.table(df)
    df = apply_probabilities(df, pred)
    final_class = get_final_class(df)
    if final_class == "decrease":
        st.info("The model predicts that the S&P 500 index will decrease more than 0.5% by the close of next trading day.")
    elif final_class == "maintain":
        st.info("The model predicts that the S&P 500 index will maintain its current level by the close of the next trading day.")
    else:
        st.info("The model predicts that the S&P 500 index will increase more than 0.5% by the close of the next trading day.")
