# main.py

# LAYER 1: DATA LAYER
# The data is fetched from RainForest API of Amazon

# Save the DataFrame to CSV
df.to_csv('patanjali_ghee_products.csv', index=False)

# NLTK Preprocessing
nltk.download('punkt')
nltk.download('stopwords')

if 'review' in df.columns and not df.empty:
    df['review'] = df['review'].str.lower()
    df['tokens'] = df['review'].apply(lambda x: word_tokenize(x) if isinstance(x, str) else [])
    stop_words = set(stopwords.words('english'))
    df['tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])
    ps = PorterStemmer()
    df['stemmed_tokens'] = df['tokens'].apply(lambda tokens: [ps.stem(word) for word in tokens])
else:
    print("Column 'review' not found in DataFrame for preprocessing.")

# LAYER 2: MODEL BUILDING AND EVALUATION
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score

# TF-IDF Vectorization
if 'review' in df.columns and not df.empty:
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_features = tfidf_vectorizer.fit_transform(df['review'].astype(str))
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    print(tfidf_df.head())

df_sentiment = pd.read_csv('patanjali_ghee_products.csv')
if 'review' in df_sentiment.columns:
    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    df_sentiment['sentiment'] = df_sentiment['review'].dropna().apply(lambda x: sentiment_pipeline(x)[0]['label'])
    print(df_sentiment[['review', 'sentiment']].head())
else:
    print("Column 'review' not found in sentiment analysis DataFrame.")

# Model Performance Evaluation
if 'true_sentiment' in df_sentiment.columns:
    valid_indices = df_sentiment['true_sentiment'].notnull() & df_sentiment['sentiment'].notnull()
    accuracy = accuracy_score(df_sentiment.loc[valid_indices, 'true_sentiment'],
                              df_sentiment.loc[valid_indices, 'sentiment'])
    print(f"Accuracy: {accuracy:.2f}")
    report = classification_report(df_sentiment.loc[valid_indices, 'true_sentiment'],
                                   df_sentiment.loc[valid_indices, 'sentiment'],
                                   target_names=['Negative', 'Neutral', 'Positive'])
    print(report)
else:
    print("Column 'true_sentiment' not found for evaluation.")

# LAYER 3: VISUALIZATION LAYER
# Save each DataFrame to CSV
df_sentiment.to_csv('D:/publication/sentiment_analysis_output.csv', index=False)

# Word Cloud Visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['review'].dropna()))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# LAYER 4: REAL-TIME DATA INTEGRATION AND DEPLOYMENT
## Real-time monitoring and alert layer

# Example of a simple monitoring function that prints alerts
def monitor_sentiment(new_reviews_stream):
    sentiment_map = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral',
        'LABEL_2': 'Positive',
    }

    # Simulated real-time monitoring logic
    for review in new_reviews_stream:
        sentiment = sentiment_pipeline(review)[0]['label']
        sentiment_label = sentiment_map.get(sentiment, 'Unknown')

        if sentiment_label == 'Negative':
            print(f'Alert: Negative sentiment detected in review - "{review}"')


# Simulate a stream of new reviews
new_reviews_stream = ['This product is awful!', 'I love it!', 'It’s okay.']
monitor_sentiment(new_reviews_stream)
