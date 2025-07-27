from review_analyzer import ReviewAnalyzer
from plotting import Visualizer

analyzer = ReviewAnalyzer()

# Train model
acc, report, cm = analyzer.load_data_and_train("IMDB_Dataset.csv")
print(report)
Visualizer.plot_confusion_matrix(cm)

# Sample test
review = input("Enter a review: ")
result = analyzer.analyze_review(review)

print("\n Review Analysis Result")
print(f"Original Review     : {result['original_review']}")
print(f"Cleaned Review      : {result['clean_review']}")
print(f"Predicted Sentiment : {result['sentiment']}")
print(f"Emotion Scores:")
print(f"  - Anger    : {result['anger']}%")
print(f"  - Joy      : {result['joy']}%")
print(f"  - Optimism : {result['optimism']}%")
print(f"  - Sadness  : {result['sadness']}%")
print(f"Dominant Emotion    : {result['dominant_emotion']}")

# File-based analysis
file_path = input("Enter a .txt file path: ")
df = analyzer.analyze_file(file_path)
if df is not None:
    # Save CSV
    df.to_csv("user_output_tagged.csv", index=False)
    # Save TXT
    analyzer.save_results_to_txt(df, file_path="user_output_tagged.txt")
    Visualizer.plot_emotion_distribution(df)
    Visualizer.plot_sentiment_distribution(df)
