
#  Sentiment and Emotion Analysis

This project performs **sentiment classification** and **emotion detection** on movie reviews using a hybrid of traditional machine learning and transformer-based deep learning models. The purpose is to not only detect whether a review is positive or negative, but also extract the emotional context (anger, joy, optimism, sadness) of each review.

Understanding user sentiment and emotion in text data is valuable for: - Product or movie feedback analysis - Customer satisfaction monitoring - Mental health insights based on language - Real-time social media analysis

This project provides both **binary sentiment analysis** and **multiclass emotion tagging**, making it suitable for applications in NLP pipelines, data annotation tasks, and content moderation.

---
## Features
- **Text Preprocessing** (Lowercasing, punctuation removal, contractions, and more)
- **Sentiment Classification** using:
  - TF-IDF + Logistic Regression (for longer texts)
  - Transformer (`distilbert-base-uncased-finetuned-sst-2-english`) pipeline (for shorter texts)
- **Emotion Detection** using `cardiffnlp/twitter-roberta-base-emotion`
- 📂 **Batch Analysis** on `.txt` files containing multiple user reviews
- 📊 **Visualizations**: Sentiment and Emotion distribution charts
- 📄 **Output Reports**: `.csv` and `.txt` formats with full analysis

---

## Dataset

- **IMDB Dataset** of 50K movie reviews (Balanced dataset with 25K positive and 25K negative)
- Used to train the Logistic Regression model

##  Models Used

### 1. **TF-IDF + Logistic Regression** (for Sentiment Classification)

- **TF-IDF**: Converts text into numerical features by measuring word importance relative to a document and the corpus.
- **Logistic Regression**: A simple and fast linear classifier that works well with high-dimensional sparse data like TF-IDF vectors.
- **Why**: This combination is lightweight, fast, and effective for large-scale text classification when interpretability is important.

### 2. **HuggingFace Transformer (Sentiment Pipeline)**

- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Why**: For short texts (e.g., “loved it!”), TF-IDF may not have enough context. Transformers handle small text sequences better with deep contextual embeddings.

### 3. **Emotion Detection Model**

- **Model**: `cardiffnlp/twitter-roberta-base-emotion`
- **Trained On**: Twitter emotion-labeled datasets
- **Emotions Detected**: Anger, Joy, Optimism, Sadness
- **Why**: Fine-tuned specifically for emotion detection in short, informal texts (like reviews or tweets). Excellent at recognizing tone.


---

## 🛠️ How It Works

### 1. **Data Loading & Cleaning**
- Load IMDB dataset from `IMDB_Dataset.csv`
- Map 'positive' → 1, 'negative' → 0
- Apply `preprocess_text()`:
  - Lowercasing
  - Contraction handling (e.g., "don't" → "do_not")
  - Remove punctuation, numbers, and extra spaces

### 2. **Sentiment Model Training**
- **TF-IDF** Vectorization with bigrams
- **Logistic Regression** classifier (`max_iter=1000`)
- Splits dataset into 80/20 train-test
- Evaluates accuracy, confusion matrix, and classification report

### 3. **Emotion Detection**
- Loads `cardiffnlp/twitter-roberta-base-emotion`
- For each review, computes:
  - Emotion probabilities: `anger`, `joy`, `optimism`, `sadness`
  - Dominant emotion based on highest score

### 4. **Smart Sentiment Prediction**
- If cleaned review has fewer than 3 words:
  - Uses HuggingFace pipeline sentiment model (`distilbert`)
- Else:
  - Uses trained TF-IDF + Logistic Regression model

### 5. **Single Review Testing**
 - User can input any review and get immediate prediction.

### 6. **Batch Analysis**
- Accepts `.txt` file with one review per line
- Applies full sentiment and emotion analysis
- Saves outputs:
  - `user_output_tagged.csv`
  - `user_output_tagged.txt`
- Displays:
  - Emotion distribution bar chart
  - Sentiment distribution line plot

---

## Example Input File

**reviews.txt**
```
The film was stunning and emotionally moving.
I hated the pacing, it was way too slow.
Good acting, but a bit predictable.
```

---

## Output Files

- **user_output_tagged.csv**: Tabular result of analysis
- **user_output_tagged.txt**: Readable format with emotion scores
- **Visualizations**: Shown via `matplotlib` (not saved)

---

## Visualizations

- 📊 **Bar Plot**: Average Emotion Distribution (anger, joy, optimism, sadness).
![](/images/emotion_bar.png)
- 📉 **Line Plot**: Sentiment Distribution on batch analysis(% positive vs negative).
![](/images/line.png)

- 📉 **Confusion Matrics**: Measure how well the classification model is performing.
![](/images/confusion_matrix.png)

---

## Benefits of This Project

- **Dual Analysis**: Both sentiment and emotion extracted for deep understanding of text.
- ⚡ **Efficient**: Lightweight logistic regression model for long reviews + transformer fallback for short texts.
- **Deep Learning Integrated**: Harnesses power of pretrained models without needing huge compute.
- **Batch & Real-Time Ready**: Works on large text files and user inputs.
- **Customizable**: You can replace models or add languages easily.
- **Practical Use Cases**:
  - Social media comment analysis
  - Customer review monitoring
  - Emotion-aware chatbots
  - Educational feedback tools

---
## 🔗 References

- [HuggingFace Transformers](https://huggingface.co/)
- [CardiffNLP Emotion Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Scikit-learn](https://scikit-learn.org/)
