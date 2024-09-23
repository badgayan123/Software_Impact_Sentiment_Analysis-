# DEPLOYMENT
# app.py

from flask import Flask, request, jsonify
from transformers import pipeline
import logging

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained RoBERTa model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Map sentiment labels to human-readable labels
sentiment_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'review' not in request.json:
        return jsonify({'error': 'No review provided'}), 400

    try:
        # Extract the review text from the request
        review = request.json['review']
        # Perform sentiment analysis
        sentiment = sentiment_pipeline(review)[0]['label']
        # Log the received review and its sentiment
        logging.info(f'Review: "{review}" - Sentiment: {sentiment_map[sentiment]}')
        # Return the human-readable sentiment
        return jsonify({'sentiment': sentiment_map[sentiment]})
    except Exception as e:
        logging.error(f'Error processing review: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

