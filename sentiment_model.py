from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_processing import TextPreprocessor

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        X_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_vec, y_train)

    def evaluate(self, X_test, y_test):
        X_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_vec)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return acc, report, cm

    def predict(self, text):
        cleaned = TextPreprocessor.clean(text)
        vec = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vec)[0]
        return "Positive" if pred == 1 else "Negative"
