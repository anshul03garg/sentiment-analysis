import pandas as pd
import random
import json
from datetime import datetime, timedelta

random.seed(42)

positive_reviews = [
    "This product is absolutely amazing, exceeded all my expectations!",
    "Fantastic quality, very happy with my purchase. Highly recommend!",
    "Great value for money, works perfectly and arrived on time.",
    "Excellent product! Will definitely buy again from this seller.",
    "Super fast delivery and the product quality is outstanding.",
    "Love this product! It works exactly as described. Very satisfied.",
    "Best purchase I made this year. Quality is top notch.",
    "Incredible experience, customer service was helpful and responsive.",
    "Very impressed with the build quality and overall performance.",
    "Five stars! This product changed my daily routine completely.",
    "Highly satisfied, packaging was great and product works flawlessly.",
    "Amazing product at an affordable price. Totally worth it!",
]

negative_reviews = [
    "Terrible product, broke after just two days of use. Very disappointed.",
    "Complete waste of money. Does not work as advertised at all.",
    "Poor quality, looks nothing like the pictures shown online.",
    "Very unhappy with this purchase. Customer service was no help.",
    "Worst product I have ever bought. Stopped working within a week.",
    "Do not buy this! Total scam, product is cheap and flimsy.",
    "Extremely disappointed. Packaging was damaged and product was broken.",
    "Awful experience from start to finish. Will never order again.",
    "Product is defective and the return process is a nightmare.",
    "Horrible quality control. Received a completely different item.",
    "Waste of time and money. Absolutely not recommended to anyone.",
    "Very bad product. Instructions were unclear and it never worked.",
]

neutral_reviews = [
    "Product is okay, nothing special but gets the job done.",
    "Average quality for the price. Expected a bit more honestly.",
    "Delivery was on time. Product is decent, not great not bad.",
    "It works fine but I have seen better products at this price.",
    "Packaging was good but the product itself is just average.",
    "Neither impressed nor disappointed. Just an ordinary product.",
    "Product matches description. Quality is standard, nothing more.",
    "Acceptable for the price range. Would not strongly recommend though.",
    "Works as expected. Not the best but not the worst either.",
    "Fairly standard product. Does what it says, nothing extra.",
]

categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Beauty", "Toys", "Automotive"]
platforms = ["Amazon", "Flipkart", "Meesho", "Myntra", "Snapdeal"]

def generate_review():
    sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.55, 0.25, 0.20])[0]
    
    if sentiment == "positive":
        base = random.choice(positive_reviews)
        rating = random.choice([4, 5])
    elif sentiment == "negative":
        base = random.choice(negative_reviews)
        rating = random.choice([1, 2])
    else:
        base = random.choice(neutral_reviews)
        rating = 3

    date = datetime.now() - timedelta(days=random.randint(1, 730))
    
    return {
        "review_id": f"REV{random.randint(100000, 999999)}",
        "review_text": base,
        "rating": rating,
        "sentiment": sentiment,
        "category": random.choice(categories),
        "platform": random.choice(platforms),
        "helpful_votes": random.randint(0, 150),
        "date": date.strftime("%Y-%m-%d"),
        "verified_purchase": random.choice([True, False])
    }

print("Generating 50,000 reviews...")
reviews = [generate_review() for _ in range(50000)]

df = pd.DataFrame(reviews)
df.to_csv("data/reviews.csv", index=False)
print(f"Dataset saved! Shape: {df.shape}")
print(df['sentiment'].value_counts())
print(df.head(3))