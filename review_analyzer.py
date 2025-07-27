import pandas as pd
from tqdm import tqdm
import os
from transformers import pipeline
from data_processing import TextPreprocessor
from sentiment_model import SentimentModel
from emotion_detector import EmotionDetector
from sklearn.model_selection import train_test_split

class ReviewAnalyzer:
    def __init__(self):
        self.sentiment_model = SentimentModel()
        self.emotion_detector = EmotionDetector()
        self.transformer_pipe = pipeline("sentiment-analysis")

    def load_data_and_train(self, csv_path):
        data = pd.read_csv(csv_path)
        data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
        data['clean_review'] = data['review'].apply(TextPreprocessor.clean)
        X, y = data['clean_review'], data['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.sentiment_model.train(X_train, y_train)
        return self.sentiment_model.evaluate(X_test, y_test)

    def smart_sentiment(self, text):
        cleaned = TextPreprocessor.clean(text)
        if len(cleaned.split()) < 3:
            result = self.transformer_pipe(cleaned)[0]['label']
            return "Positive" if result.upper() == "POSITIVE" else "Negative"
        else:
            return self.sentiment_model.predict(cleaned)

    def analyze_review(self, review):
        sentiment = self.smart_sentiment(review)
        anger, joy, optimism, sadness, top_emotion = self.emotion_detector.detect(review)
        return {
            "original_review": review,
            "clean_review": TextPreprocessor.clean(review),
            "sentiment": sentiment,
            "anger": anger,
            "joy": joy,
            "optimism": optimism,
            "sadness": sadness,
            "dominant_emotion": top_emotion
        }

    def analyze_file(self, file_path):
        if not os.path.exists(file_path):
            print("⚠️ File not found.")
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            reviews = [line.strip() for line in f if line.strip()]
        results = [self.analyze_review(r) for r in tqdm(reviews)]
        return pd.DataFrame(results)
    
    def save_results_to_txt(self, df, file_path="user_output_tagged.txt"):
        """Write a human-readable .txt report of each review’s sentiment & emotions."""
        with open(file_path, 'w', encoding='utf-8') as txt_file:
            for _, row in df.iterrows():
                txt_file.write(f"Original Review: {row['original_review']}\n")
                txt_file.write(f"Cleaned Review:  {row['clean_review']}\n")
                txt_file.write(f"Sentiment:       {row['sentiment']}\n")
                txt_file.write(f"Dominant Emotion:{row['dominant_emotion']}\n")
                txt_file.write(
                    f"Emotion Scores -> Anger: {row['anger']}%, "
                    f"Joy: {row['joy']}%, "
                    f"Optimism: {row['optimism']}%, "
                    f"Sadness: {row['sadness']}%\n"
                )
                txt_file.write("-" * 60 + "\n")
