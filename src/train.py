import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def load_data():
    print("Loading processed data...")
    df = pd.read_csv("data/processed_reviews.csv")
    df = df.dropna(subset=['processed_text', 'label'])
    print(f"Total samples: {len(df)}")
    return df

def split_data(df):
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def build_pipelines():
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    pipelines = {
        "Logistic Regression": Pipeline([
            ('tfidf', tfidf),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('clf', MultinomialNB(alpha=0.1))
        ]),
        "Linear SVM": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                      sublinear_tf=True)),
            ('clf', LinearSVC(random_state=42, max_iter=2000))
        ])
    }
    return pipelines

def plot_confusion_matrix(cm, model_name, labels):
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    plt.savefig(f"outputs/confusion_matrix_{fname}.png", dpi=150)
    plt.close()
    print(f"  Saved: outputs/confusion_matrix_{fname}.png")

def plot_model_comparison(results):
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] * 100 for n in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, accuracies, color=['#4C72B0', '#DD8452', '#55A868'],
                   edgecolor='black', width=0.5)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() - 2,
                 f'{acc:.2f}%', ha='center', va='top',
                 color='white', fontweight='bold', fontsize=12)
    plt.ylim(70, 100)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.tight_layout()
    plt.savefig("outputs/model_comparison.png", dpi=150)
    plt.close()
    print("Saved: outputs/model_comparison.png")

def train_and_evaluate(pipelines, X_train, X_test, y_train, y_test):
    label_names = ['Negative', 'Neutral', 'Positive']
    results = {}
    best_acc = 0
    best_model_name = ""

    for name, pipeline in pipelines.items():
        print(f"\nTraining: {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"  Accuracy: {acc * 100:.2f}%")
        print(f"\n{classification_report(y_test, y_pred, target_names=label_names)}")

        plot_confusion_matrix(cm, name, label_names)

        results[name] = {'accuracy': acc, 'pipeline': pipeline}

        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    return results, best_model_name

def save_best_model(results, best_model_name):
    best_pipeline = results[best_model_name]['pipeline']
    joblib.dump(best_pipeline, "models/best_model.pkl")
    print(f"\nBest Model: {best_model_name} ({results[best_model_name]['accuracy']*100:.2f}%)")
    print("Saved: models/best_model.pkl")

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipelines = build_pipelines()
    results, best_model_name = train_and_evaluate(
        pipelines, X_train, X_test, y_train, y_test
    )
    plot_model_comparison(results)
    save_best_model(results, best_model_name)
    print("\nTraining Complete!")