import streamlit as st
from backend import get_headlines, apply_probabilities, get_final_class, pred

# Main title of the app
st.title("S&P 500 Predictor Using NLP")

# Header image
st.image("header.png", use_column_width=True)

# Introduction section
st.markdown("""
    This application forecasts the movement of the S&P 500 index based on recent financial news headlines 
    from Yahoo Finance. It uses a fine-tuned transformer model to predict whether the index will:
    - Increase by more than 0.5%
    - Decrease by more than 0.5%
    - Maintain its current level

    For more details, visit the [GitHub repository](https://github.com/jeroenvdmbrugge/SP500-Predictor-WebApp).

    ---
""")

# Initialize session state
if "headlines_obtained" not in st.session_state:
    st.session_state.headlines_obtained = False

# Step 1: Fetch Headlines
st.markdown("### Step 1: Get Latest Headlines from Yahoo Finance")
num_headlines = st.slider("Select number of headlines:", min_value=1, max_value=10, value=5)
if st.button("Fetch Headlines"):
    with st.spinner("Fetching headlines..."):
        df = get_headlines(num_headlines)
        if df.empty:
            st.warning("No headlines were found. Please try again later.")
        else:
            st.session_state.df = df  # Store df in session state
            st.session_state.headlines_obtained = True

if st.session_state.headlines_obtained:
    df = st.session_state.df
    st.table(df)

# Step 2: Predict S&P 500 Movement
if st.session_state.headlines_obtained:
    st.markdown("### Step 2: Analyze Headlines to Predict S&P 500 Movement")
    if st.button("Get Prediction"):
        with st.spinner("Analyzing headlines and predicting..."):
            df = st.session_state.df
            df = apply_probabilities(df, pred)
            final_class = get_final_class(df)

            if final_class == "decrease":
                st.info(f"""
                Based on these {num_headlines} headlines, the model predicts that the S&P 500 index will decrease 
                by more than 0.5% by the close of the next trading day.
                """)

            elif final_class == "maintain":
                st.info(f"""
                Based on these {num_headlines} headlines, the model predicts that the S&P 500 index will maintain 
                its current level by the close of the next trading day.
                """)
            else:
                st.info(f"""
                Based on these {num_headlines} headlines, the model predicts that the S&P 500 index will increase 
                by more than 0.5% by the close of the next trading day.
                """)
