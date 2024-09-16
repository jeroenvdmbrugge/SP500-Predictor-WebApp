# S&P 500 Predictor Using NLP
![Header Image](header.png)

## Project Overview
This project is a web application built with Streamlit that aims to predict the movement of the S&P 500 index for the next trading day by analyzing financial news headlines using Natural Language Processing (NLP).

## How It Works
The app utilizes a [transformer model](https://huggingface.co/jeroenvdmbrugge/sp500-predictor-individual-headlines), which has been fine-tuned on a dataset of financial news headlines. The headlines in this training dataset were classified into three categories based on the change in the closing price of the S&P 500 index on the next trading day: increase, decrease, or maintain. An increase is defined as a closing price change of more than 0.5%, a decrease is a change of more than 0.5%, and maintain indicates a change within -0.5% to 0.5%.


### Application Workflow
1. **Get Latest Headlines from Yahoo Finance**:
   - Select the number of recent headlines (between 1 and 10) to use for the prediction.
   - Click "Fetch Headlines" to retrieve these headlines from Yahoo Finance.

2. **Analyze Headlines to Predict S&P 500 Movement**:
   - After fetching the headlines, click "Get Prediction" to analyze them.
   - The app analyzes each headline individually with the transformer model, and calculates the average probability for each of the three categories to provide a final prediction on whether the S&P 500 index will increase, decrease, or maintain its level by the close of the next trading day.



## Try It Out
The app is deployed on the Streamlit Community Cloud. You can try it out here: [S&P 500 Predictor Web App](https://jvdm-sp500-predictor-webapp-v1.streamlit.app).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/jeroenvdmbrugge/SP500-Predictor-WebApp.git
   ```
2. Navigate to the project directory:
   ```bash
   cd SP500-Predictor-WebApp
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Intended Uses & Limitations
This app and its underlying model are intended for experimental purposes only, and should not be used for making financial decisions. Further fine-tuning on more specific and extensive datasets will be necessary to improve its accuracy and reliability.




