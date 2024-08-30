import streamlit as st
from backend import get_headlines, apply_probabilities, get_final_class, pred

st.title("S&P500 Predictor")
st.image("header.png", use_column_width=True)
st.write("")
st.markdown("""
        This web application predicts the S&P 500 index movement for the next trading day. 
        It analyzes 10 recent headlines using a pretrained transformer model to forecast whether the index will:
        - Decrease by more than 0.5%\n
        - Maintain its current level\n
        - Increase by more than 0.5%\n
        Click the button below to get started!
         """)

if st.button('Get Headlines and Predict S&P500 Movement'):
    df = get_headlines()
    st.table(df)
    df = apply_probabilities(df, pred)
    final_class = get_final_class(df)
    if final_class == "decrease":
        st.info("Our model predicts that the S&P500 will decrease by more than 0.5% on the next trading day.")
    elif final_class == "maintain":
        st.info("Our model predicts that the S&P500 will maintain its current level on the next trading day.")
    else:
        st.info("Our model predicts that the S&P500 will increase by more than 0.5% on the next trading day.")
