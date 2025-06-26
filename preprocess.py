# data/preprocess.py

import pandas as pd
from datasets import Dataset

def load_data():
    df = pd.read_csv("C:\Users\prabh\OneDrive\Desktop\self_healing_classifier\IMDB Dataset.csv")  # Use the correct path and filename
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df["label"] = df["label"].map({"positive": 1, "negative": 0})
    return Dataset.from_pandas(df).train_test_split(test_size=0.2)
