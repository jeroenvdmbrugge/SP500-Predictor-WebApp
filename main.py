from backend import get_cnbc_headlines, apply_probabilities, get_final_class
import pandas as pd

print("S&P500 Predictor")
print("This is a web application that predicts the direction of the S&P500 on the next trading day. "
      "It analyzes 10 recent headlines from CNBC using a pre-trained transformer model to predict whether the index will"
      " decrease by more than 0.5%, a sustain its current level within 0.5%, or increase by more than 0.5%.")
print("LATEST HEADLINES")
print("Push the button to collect the latest headlines from CNBC")
df = get_cnbc_headlines()
print(df.Headlines.values)
print("PREDICTION")
print("Push the button to predict the direction of the S&P500 on the next trading day")
df = apply_probabilities(df)
#print(df.head())
final_class = get_final_class(df)
print(f"Our model predicts that the S&P500 will {final_class.lower()} on the next trading day.")
#print(final_class)
