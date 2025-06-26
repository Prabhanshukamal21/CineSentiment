import gradio as gr
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import numpy as np

# Load model
model_path = r"C:\Users\prabh\OneDrive\Desktop\self_healing_classifier\model_output"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    label = torch.argmax(probs).item()
    sentiment = "Positive" if label == 1 else "Negative"
    confidence = f"{probs[0][label].item():.2%}"
    return f"Sentiment: {sentiment}\nConfidence: {confidence}"

# Gradio UI
gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter a movie review..."),
    outputs="text",
    title="IMDB Movie Review Sentiment Classifier",
    description="Enter a movie review and see if it's predicted as Positive or Negative."
).launch()
