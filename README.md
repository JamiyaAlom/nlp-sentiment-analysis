
# 🧠 Sentiment and Emotion Analysis on IMDB Reviews

This project performs **sentiment classification** and **emotion detection** on movie reviews using a hybrid of traditional machine learning and transformer-based deep learning models. The purpose is to not only detect whether a review is positive or negative, but also extract the emotional context (anger, joy, optimism, sadness) of each review.

---

## 🚀 Features

- 🔤 **Text Preprocessing** (Lowercasing, punctuation removal, contractions, and more)
- 🧠 **Sentiment Classification** using:
  - TF-IDF + Logistic Regression (for longer texts)
  - Transformer (`distilbert-base-uncased-finetuned-sst-2-english`) pipeline (for shorter texts)
- ❤️ **Emotion Detection** using `cardiffnlp/twitter-roberta-base-emotion`
- 📂 **Batch Analysis** on `.txt` files containing multiple user reviews
- 📊 **Visualizations**: Sentiment and Emotion distribution charts
- 📄 **Output Reports**: `.csv` and `.txt` formats with full analysis

---

## 📦 Installation

Before running the script, install all necessary dependencies:

```bash
pip install pandas scikit-learn matplotlib tqdm transformers torch
```

Or use a `requirements.txt` if available:

```bash
pip install -r requirements.txt
```

---

## 🧾 Dataset

- **IMDB Dataset** of 50K movie reviews (Balanced dataset with 25K positive and 25K negative)
- Used to train the Logistic Regression model

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

### 5. **Interactive Input**
- Prompts user for a review
- Displays:
  - Cleaned review
  - Predicted sentiment
  - Emotion scores and dominant emotion

### 6. **Batch Input from File**
- Accepts `.txt` file with one review per line
- Applies full sentiment and emotion analysis
- Saves outputs:
  - `user_output_tagged.csv`
  - `user_output_tagged.txt`
- Displays:
  - Emotion distribution bar chart
  - Sentiment distribution line plot

---

## 📂 Example Input File

**reviews.txt**
```
The film was stunning and emotionally moving.
I hated the pacing, it was way too slow.
Good acting, but a bit predictable.
```

---

## 💾 Output Files

- **user_output_tagged.csv**: Tabular result of analysis
- **user_output_tagged.txt**: Readable format with emotion scores
- **Visualizations**: Shown via `matplotlib` (not saved)

---

## 📈 Visualizations

- 📊 **Bar Plot**: Average Emotion Distribution (anger, joy, optimism, sadness)
- 📉 **Line Plot**: Sentiment Distribution (% positive vs negative)

---

## ✅ How to Run

```bash
python sentiment_emotion_analysis.py
```

Then follow the interactive prompts for either:
- Custom review input
- Input text file analysis

---

## 📁 Project Files

```
📦 sentiment_emotion_analysis
├── IMDB_Dataset.csv                # Input training dataset
├── sentiment_emotion_analysis.py  # Main analysis script
├── user_output_tagged.csv         # Output from user file (CSV)
├── user_output_tagged.txt         # Output from user file (TXT)
└── README.md                      # This file
```

---

## 📌 Notes

- Make sure to download the `IMDB_Dataset.csv` file beforehand.
- Internet connection required to download pretrained transformer models.
- GPU acceleration recommended but not required.

---

## 🙋 Author

**Jamiya Alom**

For questions, feel free to reach out or create a GitHub issue.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🔗 References

- [HuggingFace Transformers](https://huggingface.co/)
- [CardiffNLP Emotion Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Scikit-learn](https://scikit-learn.org/)
