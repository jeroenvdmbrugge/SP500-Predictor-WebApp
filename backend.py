import newspaper
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

def get_cnbc_headlines():
    cnbc = newspaper.build("https://www.cnbc.com/us-economy/", memoize_articles=False)
    df = pd.DataFrame()

    for article in cnbc.articles[0:10]:
        article.download()
        article.parse()

        temp_df = pd.DataFrame([{"Headlines": article.title, "Date": article.publish_date}])
        df = pd.concat([df, temp_df], ignore_index=True)

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
    df = get_cnbc_headlines()
    print(df.head())
    df = apply_probabilities(df)
    print(df.head())
    final_class = get_final_class(df)
    print(final_class)
