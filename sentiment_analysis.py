import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Load Data & Train Improved Sentiment Model ----------------
data = pd.read_csv('IMDB_Dataset.csv')
#data

# ---------------- Text Preprocessing ----------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"\b(can|does|do|did|is|are|was|were|would|should|could|have|has|had)\s+not\b", r"\1_not", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
data['clean_review'] = data['review'].apply(preprocess_text)

# ---------------- Emotion Detection Setup ----------------
model_name = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_emotion = AutoModelForSequenceClassification.from_pretrained(model_name)
emotion_labels = ['anger', 'joy', 'optimism', 'sadness']

def get_emotion(text):
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series([0.0, 0.0, 0.0, 0.0, "neutral"])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model_emotion(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
    emotion_probs = {emotion_labels[i]: round(float(probs[i]) * 100, 2) for i in range(len(emotion_labels))}
    top_emotion = max(emotion_probs, key=emotion_probs.get)
    return pd.Series([emotion_probs['anger'], emotion_probs['joy'], emotion_probs['optimism'], emotion_probs['sadness'], top_emotion]) 

X = data['clean_review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

y_train_pred = model.predict(X_train_vec)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_accuracy * 100, test_accuracy * 100], color=['green', 'blue'])
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Training vs Testing Accuracy')
for i, acc in enumerate([train_accuracy, test_accuracy]):
    plt.text(i, acc * 100 + 1, f'{acc * 100:.2f}%', ha='center')
plt.tight_layout()
plt.show()

print("------------ Model Training Summary -------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
accuracy = np.trace(cm) / np.sum(cm)
print(f"The accuracy of our model is {accuracy * 100:.2f}%, which means {accuracy * 100:.2f}% of the predictions are accurate.")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Logistic Regression Sentiment Classifier")
plt.show()

# ---------------- Transformer pipeline for short texts ----------------
transformer_sentiment = pipeline("sentiment-analysis")

# ---------------- Sentiment Prediction ----------------
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"

def smart_sentiment_prediction(text):
    cleaned = preprocess_text(text)
    if len(cleaned.split()) < 3 and cleaned != "":
        # Use transformer sentiment model for short texts
        result = transformer_sentiment(cleaned)[0]['label']
        return "Positive" if result.upper() == 'POSITIVE' else "Negative"
    else:
        return predict_sentiment(text)

# ---------------- Test sample input ----------------
sample_review = input("Enter a review to predict sentiment: ")
print("\nSample Review (cleaned):", preprocess_text(sample_review))
processed_text = smart_sentiment_prediction(sample_review)
anger, joy, optimism, sadness, top_emotion = get_emotion(processed_text)
print("\nPredicted Sentiment:", smart_sentiment_prediction(sample_review))
print(f"anger:{anger}, joy:{joy}, optimism:{optimism}, sadness:{sadness}")
print(f"Domination Emotion: {top_emotion}")


# ---------------- User Input File Analysis ----------------
print("\n📢 User input file Analysis with sentiment and emotions.")
print("Please provide a file (.txt) with one review per line.")
user_file = input("Enter the path or name of your input file (e.g., reviews.txt): ").strip()

if os.path.exists(user_file):
    print("✅ File found. Starting line-by-line sentiment and emotion tagging...")

    reviews = []
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            review = line.strip()
            if review:
                reviews.append(review)

    result_data = []
    tqdm.pandas(desc="Processing Reviews")
    for review in tqdm(reviews):
        clean = preprocess_text(review)
        sentiment = smart_sentiment_prediction(clean)
        anger, joy, optimism, sadness, top_emotion = get_emotion(clean)
        result_data.append({
            "original_review": review,
            "clean_review": clean,
            "sentiment": sentiment,
            "anger": anger,
            "joy": joy,
            "optimism": optimism,
            "sadness": sadness,
            "dominant_emotion": top_emotion
        })

    result_df = pd.DataFrame(result_data)
    # Save to CSV
    result_df.to_csv('user_output_tagged.csv', index=False)



        # Save to TXT
    with open('user_output_tagged.txt', 'w', encoding='utf-8') as txt_file:
        for _, row in result_df.iterrows():
            txt_file.write(f"Original Review: {row['original_review']}\n")
            txt_file.write(f"Cleaned Review: {row['clean_review']}\n")
            txt_file.write(f"Sentiment: {row['sentiment']}\n")
            txt_file.write(f"Dominant Emotion: {row['dominant_emotion']}\n")
            txt_file.write(f"Emotion Scores -> Anger: {row['anger']}%, Joy: {row['joy']}%, Optimism: {row['optimism']}%, Sadness: {row['sadness']}%\n")
            txt_file.write("-" * 60 + "\n")

    
    print("✅ Line-by-line analysis complete.")
    print("📁 Results saved to:")
    print("   - user_output_tagged.csv")
    print("   - user_output_tagged.txt")

    # ---------------- Visualizations ----------------
    # Average Emotion distribution line plot
    avg_emotions = result_df[['anger', 'joy', 'optimism', 'sadness']].mean()
    avg_emotions.plot(kind='bar', color='skyblue')
    plt.title("Average Emotion Distribution")
    plt.ylabel("Percentage")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Average Sentiment distribution line plot
    sentiment_counts = result_df['sentiment'].value_counts(normalize=True).sort_index()
    sentiment_labels = ['Negative', 'Positive']
    sentiment_values = [sentiment_counts.get('Negative',0)*100, sentiment_counts.get('Positive',0)*100]
    plt.figure(figsize=(6,4))
    plt.plot(sentiment_labels, sentiment_values, marker='o', linestyle='-', color='tab:green')
    plt.title("Sentiment Distribution (%)")
    plt.ylabel("Percentage")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print("⚠️ File not found. Please check the path and try again.")

# ---------------- End of Script ----------------


    