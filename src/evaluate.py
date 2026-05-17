import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import cross_val_score
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']

def load_model_and_data():
    print("Loading model and data...")
    model = joblib.load("models/best_model.pkl")
    df = pd.read_csv("data/processed_reviews.csv")
    df = df.dropna(subset=['processed_text', 'label'])
    return model, df

def evaluate_on_test(model, df):
    from sklearn.model_selection import train_test_split
    X = df['processed_text']
    y = df['label']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n--- Model Evaluation on Test Set ---")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
    return X_test, y_test, y_pred

def plot_sentiment_distribution(df):
    counts = df['sentiment'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black')
    axes[0].set_title('Sentiment Distribution (Bar)', fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, str(v), ha='center', fontweight='bold')

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=colors, startangle=140)
    axes[1].set_title('Sentiment Distribution (Pie)', fontweight='bold')

    plt.tight_layout()
    plt.savefig("outputs/sentiment_distribution.png", dpi=150)
    plt.close()
    print("Saved: outputs/sentiment_distribution.png")

def plot_rating_vs_sentiment(df):
    plt.figure(figsize=(9, 5))
    ct = pd.crosstab(df['rating'], df['sentiment'])
    ct.plot(kind='bar', ax=plt.gca(),
            color=['#e74c3c', '#f39c12', '#2ecc71'],
            edgecolor='black')
    plt.title('Rating vs Sentiment', fontweight='bold')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig("outputs/rating_vs_sentiment.png", dpi=150)
    plt.close()
    print("Saved: outputs/rating_vs_sentiment.png")

def plot_category_sentiment(df):
    plt.figure(figsize=(12, 6))
    ct = pd.crosstab(df['category'], df['sentiment'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True,
            color=['#e74c3c', '#f39c12', '#2ecc71'],
            edgecolor='black', ax=plt.gca())
    plt.title('Sentiment % by Product Category', fontweight='bold')
    plt.xlabel('Category')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentiment', bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig("outputs/category_sentiment.png", dpi=150)
    plt.close()
    print("Saved: outputs/category_sentiment.png")

def generate_wordclouds(df):
    sentiments = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (sentiment, color) in zip(axes, sentiments.items()):
        text = ' '.join(df[df['sentiment'] == sentiment]['processed_text'].dropna())
        if not text.strip():
            continue
        wc = WordCloud(width=500, height=350,
                       background_color='white',
                       colormap='RdYlGn',
                       max_words=100).generate(text)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{sentiment.capitalize()} Reviews', fontsize=14,
                     fontweight='bold', color=color)

    plt.suptitle('Word Clouds by Sentiment', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/wordclouds.png", dpi=150)
    plt.close()
    print("Saved: outputs/wordclouds.png")

def plot_monthly_trend(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
    monthly.index = monthly.index.astype(str)

    plt.figure(figsize=(14, 5))
    for sentiment, color in zip(['positive', 'negative', 'neutral'],
                                 ['#2ecc71', '#e74c3c', '#f39c12']):
        if sentiment in monthly.columns:
            plt.plot(monthly.index, monthly[sentiment],
                     label=sentiment.capitalize(), color=color, linewidth=2)

    plt.title('Monthly Sentiment Trend', fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Review Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/monthly_trend.png", dpi=150)
    plt.close()
    print("Saved: outputs/monthly_trend.png")

def test_custom_reviews(model):
    test_reviews = [
        "This product is absolutely fantastic, best purchase ever!",
        "Terrible quality, broke after one day. Complete waste of money.",
        "It is okay, nothing special. Average product for the price.",
        "Exceeded all my expectations. Highly recommend to everyone!",
        "Very disappointed with the purchase. Would not buy again.",
    ]

    print("\n--- Custom Review Predictions ---")
    for review in test_reviews:
        pred = model.predict([review])[0]
        label = LABEL_MAP[pred]
        print(f"  [{label:8s}] {review[:65]}...")

if __name__ == "__main__":
    model, df = load_model_and_data()

    X_test, y_test, y_pred = evaluate_on_test(model, df)

    print("\nGenerating visualizations...")
    plot_sentiment_distribution(df)
    plot_rating_vs_sentiment(df)
    #plot_category_sentiment(df)
    generate_wordclouds(df)
    #plot_monthly_trend(df)

    test_custom_reviews(model)

    print("\nAll outputs saved in outputs/ folder!")