import FinNews as fn
import pandas as pd
from datetime import date, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch


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


def get_probabilities(headline):
    tokenizer = AutoTokenizer.from_pretrained("jvdm_sp500_dr_individual_v1")
    model = AutoModelForSequenceClassification.from_pretrained("jvdm_sp500_dr_individual_v1")
    pred = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)

    result = pred(headline)
    scores = [label["score"] for label in result[0]]

    return torch.tensor(scores)


def apply_probabilities(df):
    df["Probabilities"] = df["Headlines"].apply(get_probabilities)

    return df

def get_final_class(df):
    prob_tensors = torch.stack(df["Probabilities"].tolist())
    average_probabilities = prob_tensors.mean(dim=0)
    class_labels = ['Decrease', 'Sustain', 'Increase']
    final_class = class_labels[torch.argmax(average_probabilities).item()]

    return final_class


if __name__ == "__main__":
    df = get_headlines()
    print(df.head())
    df = apply_probabilities(df)
    print(df.head())
    final_class = get_final_class(df)
    print(final_class)
