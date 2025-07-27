import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class EmotionDetector:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-emotion"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['anger', 'joy', 'optimism', 'sadness']

    def detect(self, text):
        if not isinstance(text, str) or not text.strip():
            return [0.0]*4 + ["neutral"]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
        emotion_scores = [round(float(p) * 100, 2) for p in probs]
        top_emotion = self.labels[np.argmax(probs)]
        return emotion_scores + [top_emotion]
