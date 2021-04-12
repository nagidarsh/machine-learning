# -*- coding: utf-8 -*-
"""
@author - Darshjyot
"""

#import libraries
import numpy as np
import pandas as pd
from surprise import SVD
import pickle
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')


# Loading Models and vectorizer
recommender_model = pickle.load(open('model/recommend_engine.pkl', 'rb'))
sentiment_model = pickle.load(open('model/sentimental_model.pkl', 'rb'))
count_vect = pickle.load(open('model/count_vect.pkl', 'rb'))

# Loading ratings and product sentiments
ratings = pd.read_csv('dataset/ratings.csv')
ps = pd.read_csv('dataset/product_sentiments.csv')

# Set index for faster search
ratings.set_index('reviews_username', inplace=True)




def predict(username, ntop = 5):
  """Predicts the top n products for given user"""
  product_list = ratings.loc[username]
  product_ids = pd.DataFrame(columns=['id', 'rating'])

  for index, value in product_list.items():
    # pick product only if it is not reviewed by the user
    if np.isnan([value]):
      # predict rating and append to list
      product_ids = product_ids.append({'id': index, 'rating': 
                          recommender_model.predict(username, index).est}, ignore_index = True)
  
  return _predict_sentiment(product_ids.sort_values(by='rating', ascending=False).head(max(20, ntop)), ntop)




def _predict_sentiment(df, ntop = 5):
  """Filters and ranks selected products using sentiments"""
  sentiments = ps[ps.id.isin(df['id'].to_list())] # Filter reivews for only top 20/n recommended items 
  sentiments['sentiment'] = _get_sentiment(sentiments['clean_text'])

  # Sorting based on total postive reviews
  grouped_df = sentiments.groupby('id')['sentiment'].sum().sort_values(ascending=False)
  grouped_df = pd.DataFrame(grouped_df).reset_index()

  return grouped_df[['id']].head(ntop)




def _get_sentiment(txt):
  """get sentiment for given text data"""
  input_vect = count_vect.transform(txt) #Convert text to input for Model

  #Predict
  return sentiment_model.predict(input_vect)[0]