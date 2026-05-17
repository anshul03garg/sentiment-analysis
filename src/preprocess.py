import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

tqdm.pandas()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocess_pipeline(text):
    text = clean_text(text)
    tokens = tokenize_and_remove_stopwords(text)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)

def preprocess_dataframe(df):
    print("Step 1: Cleaning text...")
    df['cleaned_text'] = df['review_text'].progress_apply(clean_text)

    print("Step 2: Tokenizing + removing stopwords...")
    df['tokens'] = df['cleaned_text'].progress_apply(tokenize_and_remove_stopwords)

    print("Step 3: Lemmatizing...")
    df['processed_text'] = df['tokens'].progress_apply(
        lambda tokens: ' '.join(lemmatize_tokens(tokens))
    )

    print("Step 4: Encoding labels...")
    label_map = {'positive': 2, 'neutral': 1, 'negative': 0}
    df['label'] = df['sentiment'].map(label_map)

    print(f"\nPreprocessing complete!")
    print(f"Total reviews: {len(df)}")
    print(f"Null values: {df['processed_text'].isnull().sum()}")
    print(f"\nSentiment Distribution:")
    print(df['sentiment'].value_counts())

    return df

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv("data/reviews.csv")

    df = preprocess_dataframe(df)

    df.to_csv("data/processed_reviews.csv", index=False)
    print("\nSaved: data/processed_reviews.csv")

    print("\nSample processed reviews:")
    print(df[['review_text', 'processed_text', 'sentiment']].head(5).to_string())