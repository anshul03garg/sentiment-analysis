from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model
model = joblib.load("models/best_model.pkl")

# NLP setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
EMOJI_MAP = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😠'}
COLOR_MAP = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    raw_text = data['text'].strip()
    if len(raw_text) < 3:
        return jsonify({'error': 'Text too short'}), 400

    processed = preprocess(raw_text)
    prediction = model.predict([processed])[0]
    label = LABEL_MAP[prediction]

    # Confidence scores
    try:
        proba = model.predict_proba([processed])[0]
        confidence = {LABEL_MAP[i]: round(float(p) * 100, 2)
                      for i, p in enumerate(proba)}
    except AttributeError:
        confidence = {label: 100.0}

    return jsonify({
        'text'      : raw_text,
        'sentiment' : label,
        'emoji'     : EMOJI_MAP[label],
        'color'     : COLOR_MAP[label],
        'confidence': confidence
    })

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'No texts provided'}), 400

    results = []
    for text in data['texts'][:50]:  # max 50
        processed = preprocess(text)
        pred = model.predict([processed])[0]
        label = LABEL_MAP[pred]
        results.append({
            'text'     : text[:100],
            'sentiment': label,
            'emoji'    : EMOJI_MAP[label],
            'color'    : COLOR_MAP[label]
        })

    summary = {
        'total'   : len(results),
        'positive': sum(1 for r in results if r['sentiment'] == 'Positive'),
        'neutral' : sum(1 for r in results if r['sentiment'] == 'Neutral'),
        'negative': sum(1 for r in results if r['sentiment'] == 'Negative'),
    }

    return jsonify({'results': results, 'summary': summary})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)