SentimentViz accelerator uses Flask and the Hugging Face Transformers library. It leverages a pre-trained RoBERTa model to classify the sentiment of text reviews into three categories: Negative, Neutral, and Positive. It is structured into multiple layers, each responsible for a specific functionality, from data handling to real-time deployment.

Code Structure


Layer 1: Data Layer
Purpose: This layer is responsible for fetching product data from the RainForest API, preprocessing the review text, and preparing it for analysis.
Key Functions:
Data Fetching: Retrieves product details and reviews from the API.
NLTK Preprocessing: Cleans and tokenizes the review text, removing stopwords and applying stemming.

Layer 2: Model Building and Evaluation
Purpose: This layer handles the model training and evaluation process.
Key Functions:
TF-IDF Vectorization: Converts the text data into a numerical format for model input.
Sentiment Analysis: The Hugging Face sentiment analysis pipeline is used to predict sentiment labels.
Performance Evaluation: Calculates the accuracy of the sentiment predictions against accurate labels, providing a classification report.

Layer 3: Visualization Layer
Purpose: This layer generates visualizations representing the data and sentiment analysis results.
Key Functions:
Word Cloud Generation: Creates a visual representation of frequently used words in reviews.
Data Export: Saves processed data and sentiment analysis results to CSV files for further examination.

Layer 4: Real-Time Data Integration and Deployment
Purpose: This layer implements the Flask application that serves the sentiment analysis API.
Key Functions:
API Endpoint: Provides a POST endpoint (/predict) to analyze the sentiment of incoming reviews.
Real-Time Monitoring: Monitors a stream of new reviews for negative sentiment alerts.

Docker
A Dockerfile is included for easy application deployment. It sets up the environment and installs necessary dependencies, allowing the application to run in a containerized environment.
Usage
To run the application locally:
Clone the repository and install the required packages.
Use the command python app.py to start the API server.
Build and run the Docker image for containerized deployment as outlined in the Docker section.

Usage
To run the application locally:
Clone the repository and install the required packages.
Use the command python app.py to start the API server.
Build and run the Docker image for containerized deployment as outlined in the Docker section.

Contribution
Contributions are welcome! Feel free to fork the repository, create a feature branch, and submit a pull request.
