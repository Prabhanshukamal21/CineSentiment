README.md

# README.md
"""
# Self-Healing Text Classifier (LangGraph)

This project fine-tunes a transformer to classify text and uses a LangGraph DAG to self-correct low-confidence predictions.

## ✅ Features
- Fine-tuned sentiment model (IMDb dataset)
- LangGraph pipeline: inference → confidence check → fallback
- CLI interface
- Logging of predictions and fallback interactions

## 🛠️ How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Fine-tune the model: `code/model_train.py`
4. web application: `python model/app.py`

## 💡 Example
```
Enter your text: The movie was painfully slow.
[InferenceNode] Predicted: Positive | Confidence: 54%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Was this a negative review?
User: Yes
✅ Final Label: Negative
```

## 📁 Logs
All predictions and fallbacks are logged in `.gradio` file
