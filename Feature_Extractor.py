import numpy as np

import spacy
from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from textblob import TextBlob

from Helper_Feature_Extractor import Helper_FeatureExtraction
from Preprocessing import Preprocessing


class FeatureExtraction:

    def __init__(self):
        self.tweetsPrp = Preprocessing()
        self.nlp = spacy.load('en')
        self.df = self.tweetsPrp.load_input_feature_extraction()
        self.hepler_fe = Helper_FeatureExtraction()
        self.norm_df = self.create_dataframe_for_normalized_tweets()
        self.tfidf_feature = self.tfidf_from_tweets()
        self.bow_feature = self.bow_features_from_tweets()

    def get_dataframe_for_normalized_tweets(self):
        return self.norm_df

    def create_dataframe_for_normalized_tweets(self):
        df = self.tweetsPrp.load_input_feature_extraction()
        df.dropna(subset=['text'], how='all', inplace=True) # drop missing values
        le = preprocessing.LabelEncoder() # replace categorical data in 'categories' with numerical value
        df['categories'] = le.fit_transform(df['categories'])
        #normalized_tweets = self.hepler_fe.extract_keywords_from_tweets(self.df)
        normalized_tweets = self.hepler_fe.include_indicatorTerms_in_tweets(df)
        new_col = np.asanyarray(normalized_tweets)
        df['norm_tweets'] = new_col
        return df

    def tfidf_from_tweets(self):
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), use_idf= True)
        feature_matrix = tfidf.fit_transform(self.norm_df['norm_tweets'])
        return feature_matrix

    def bow_features_from_tweets(self):
        count_vec = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        feature_matrix = count_vec.fit_transform(self.norm_df['norm_tweets'])
        #------------------- using TF-IDF transformer over count_vectorizer--------------
        # x_train_counts = count_vec.fit_transform(self.df['norm_tweets'])
        #tfidf_transformer = TfidfTransformer()
        #feature_matrix = tfidf_transformer.fit_transform(x_train_counts)

        return feature_matrix

    def sentiment_features_from_tweets(self):

        self.norm_df['sentiment'] = self.norm_df['text'].apply(
            lambda tweet: TextBlob(tweet).polarity)  # find sentiment scores by Textblob

        return self.norm_df['sentiment']
