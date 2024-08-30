import FinNews as fn
import pandas as pd
from datetime import date, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


tokenizer = AutoTokenizer.from_pretrained("jeroenvdmbrugge/sp500-predictor-individual-headlines")
model = AutoModelForSequenceClassification.from_pretrained("jeroenvdmbrugge/sp500-predictor-individual-headlines")
pred = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

def get_headlines():
    yahoo = fn.Yahoo(topics=["*"])
    yahoo.get_news()
    df = yahoo.to_pandas()
    df = df[["title", "published"]]
    df["published"] = pd.to_datetime(df["published"])
    df["published"] = df["published"].dt.tz_localize(None).dt.date
    today = date.today()
    if df[df["published"] == today].empty:
        df = df[df["published"] == today - timedelta(days=1)]
    else:
        df = df[df["published"] == today]
    df.columns = ["Headlines", "Date"]
    df = df[~df["Headlines"].str.startswith(("Market Update", "Analyst Report"))]
    df = df.head(10).reset_index(drop=True)
    return df

def get_probabilities(headline, pred):
    result = pred(headline)
    scores = [label["score"] for label in result[0]]
    return torch.tensor(scores)

def apply_probabilities(df, pred):
    df["Probabilities"] = df["Headlines"].apply(lambda x: get_probabilities(x, pred))
    return df

def get_final_class(df):
    prob_tensors = torch.stack(df["Probabilities"].tolist())
    average_probabilities = prob_tensors.mean(dim=0)
    class_labels = ['decrease', 'maintain', 'increase']
    final_class = class_labels[torch.argmax(average_probabilities).item()]
    return final_class

if __name__ == "__main__":
    df = get_headlines()
    print(df.head())
    df = apply_probabilities(df, pred)
    print(df.head())
    final_class = get_final_class(df)
    print(final_class)
