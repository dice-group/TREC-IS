from Preprocessing import Preprocessing
from sklearn import preprocessing
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from Helper_Feature_Extractor import Helper_FeatureExtraction

class FeatureExtraction:

    def __init__(self):
        self.tweetsPrp = Preprocessing()
        self.df = self.tweetsPrp.load_input_feature_extraction()
        self.nlp = spacy.load('en')
        self.hepler_fe = Helper_FeatureExtraction()

    def create_dataframe_for_normalized_tweets(self):
        self.df.dropna(subset=['text'], how='all', inplace=True) # drop missing values
        le = preprocessing.LabelEncoder() # replace categorical data in 'categories' with numerical value
        self.df['categories'] = le.fit_transform(self.df['categories'])
        #normalized_tweets = self.hepler_fe.extract_keywords_from_tweets(self.df)
        normalized_tweets = self.hepler_fe.include_indicatorTerms_in_tweets(self.df)
        new_col = np.asanyarray(normalized_tweets)
        self.df['norm_tweets'] = new_col
        return self.df

    def tfidf_from_tweets(self):
        self.df = self.create_dataframe_for_normalized_tweets()
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), use_idf= True)
        feature_matrix = tfidf.fit_transform(self.df['norm_tweets'])
        return feature_matrix

    def bow_features_from_tweets(self):
        self.df = self.create_dataframe_for_normalized_tweets()
        bow = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        feature_matrix = bow.fit_transform(self.df['norm_tweets'])
        return feature_matrix
