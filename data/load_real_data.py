import pandas as pd

print("Loading Amazon Reviews dataset...")
df = pd.read_csv("data/amazon_raw.csv")

print(f"Original shape: {df.shape}")
print(df.columns.tolist())
print(df.head(3))

def get_sentiment(score):
    if score >= 4:
        return 'positive'
    elif score == 3:
        return 'neutral'
    else:
        return 'negative'

df = df[['Text', 'Score']].dropna()
df.columns = ['review_text', 'rating']
df['sentiment'] = df['rating'].apply(get_sentiment)

df = df.sample(n=min(50000, len(df)), random_state=42).reset_index(drop=True)

print(f"\nFinal shape: {df.shape}")
print(df['sentiment'].value_counts())

df.to_csv("data/reviews.csv", index=False)
print("\nSaved: data/reviews.csv ✅")