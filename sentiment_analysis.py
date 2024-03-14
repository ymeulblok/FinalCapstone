# Import Dataset with pandas
import pandas as pd
amazon_product_reviews_df = pd.read_csv('amazon_product_reviews.csv', low_memory=False)

# Import Spacy and Language Model
import spacy
nlp = spacy.load('en_core_web_sm')

# Preprocessing
def preprocess(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Remove missing values from column
amazon_product_reviews_df.dropna(subset=['reviews.text'], inplace=True)

# Apply preprocessing to each entry in the specified column
amazon_product_reviews_df['reviews.text'] = amazon_product_reviews_df['reviews.text'].apply(preprocess)

# Creating funttion for sentiment analysis
def analyze_polarity(text):
    
    # Analyze sentiment with TextBlob
    from textblob import TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    return polarity

# Test model on sample product reviews

import random

# Specify the column containing the text data you want to sample
text_column = "reviews.text"

# Get a random index within the range of the DataFrame
random_index = random.randint(0, len(amazon_product_reviews_df) - 1)

# Get the random text from the specified column at the randomly selected index
random_text = amazon_product_reviews_df.loc[random_index, text_column]
polarity_score = analyze_polarity(random_text)

if polarity_score > 0:
    sentiment = 'positive'
elif polarity_score < 0:
    sentiment = 'negative'
else:
    sentiment = 'neutral'

print(f"Text: {random_text}\nPolarity score: {polarity_score}\nSentiment: {sentiment}")





