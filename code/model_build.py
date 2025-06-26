from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split

# ✅ Load CSV from Google Drive path
df = pd.read_csv("C:\Users\prabh\OneDrive\Desktop\self_healing_classifier\IMDB Dataset.csv")
df = df.dropna()

# Encode labels
df['labels'] = df['sentiment'].map({"positive": 1, "negative": 0})

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(df['review'], df['labels'], test_size=0.2)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "labels": train_labels.tolist()})
test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "labels": test_labels.tolist()})

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# ✅ Manually define training args (no evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
)

# ✅ Custom metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Trainer (without eval strategy)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ✅ Train manually — evaluation done post-training
trainer.train()

# ✅ Evaluate manually
eval_result = trainer.evaluate()
print("✅ Evaluation:", eval_result)

# ✅ Save model
model.save_pretrained("/content/drive/MyDrive/model_output/")
tokenizer.save_pretrained("/content/drive/MyDrive/model_output/")
