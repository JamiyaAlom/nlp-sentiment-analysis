import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class Visualizer:
    @staticmethod
    def plot_confusion_matrix(cm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()

    @staticmethod
    def plot_emotion_distribution(df):
        avg = df[['anger', 'joy', 'optimism', 'sadness']].mean()
        avg.plot(kind='bar', color='skyblue')
        plt.title("Average Emotion Distribution")
        plt.ylabel("Percentage")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sentiment_distribution(df):
        counts = df['sentiment'].value_counts(normalize=True)
        labels = ['Negative', 'Positive']
        values = [counts.get('Negative', 0)*100, counts.get('Positive', 0)*100]
        plt.plot(labels, values, marker='o', color='green')
        plt.title("Sentiment Distribution")
        plt.ylabel("Percentage")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
