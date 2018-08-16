import numpy as np
import spacy
from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import TweetTokenizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
        self.tfidf_feature, self.tfidf = self.tfidf_from_tweets()
        self.bow_feature = self.bow_features_from_tweets()

    def get_dataframe_for_normalized_tweets(self):
        return self.norm_df

    def create_dataframe_for_normalized_tweets(self):
        df = self.tweetsPrp.load_input_feature_extraction()
        df.dropna(subset=['text'], how='all', inplace=True)  # drop missing values
        le = preprocessing.LabelEncoder()  # replace categorical data in 'categories' with numerical value
        df['categories'] = le.fit_transform(df['categories'])
        # normalized_tweets = self.hepler_fe.extract_keywords_from_tweets(self.df)
        normalized_tweets = self.hepler_fe.include_indicatorTerms_in_tweets(df)
        new_col = np.asanyarray(normalized_tweets)
        df['norm_tweets'] = new_col
        return df

    def tfidf_from_tweets(self):
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), use_idf=True)
        feature_matrix = tfidf.fit_transform(self.norm_df['norm_tweets'])
        return feature_matrix, tfidf

    def bow_features_from_tweets(self):
        count_vec = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        feature_matrix = count_vec.fit_transform(self.norm_df['norm_tweets'])
        # ------------------- using TF-IDF transformer over count_vectorizer--------------
        # x_train_counts = count_vec.fit_transform(self.df['norm_tweets'])
        # tfidf_transformer = TfidfTransformer()
        # feature_matrix = tfidf_transformer.fit_transform(x_train_counts)

        return feature_matrix

    def sentiment_features_from_tweets(self):
        self.norm_df['sentiment'] = self.norm_df['text'].apply(
            lambda tweet: TextBlob(tweet).polarity)  # find sentiment scores by Textblob

    def word2vec_feature_from_tweets(self):
        # loading pre-trained word embedding model on stanford datasets (sentiment140 tweets)
        word2vec = KeyedVectors.load_word2vec_format('data/word2vec_twitter_model.bin', unicode_errors='ignore',
                                                     binary=True)

        # get TFIDf for each word
        tfidf = dict(zip(self.tfidf.get_feature_names(), self.tfidf.idf_))

        def build_average_Word2vec(tokens, size):
            '''
              computing average of weighted word2vec for each tweet.
              :return:
              '''
            vec = np.zeros(size).reshape((1, size))
            count = 0.
            for word in tokens:
                try:
                    vec += word2vec[word].reshape((1, size)) * tfidf[
                        word]  # each word vector is multiplied by word's importance (tfidf)
                    count += 1.
                except KeyError:  # handling the case where the token is not
                    #  in the corpus. useful for testing.
                    continue
            if count != 0:
                vec /= count
            return vec

        tokenizer = TweetTokenizer()  # tweets tokenizer by nltk
        self.norm_df['tweetsEmbedding'] = self.norm_df['norm_tweets'].apply(
            lambda tweet: build_average_Word2vec(tokens=tokenizer.tokenize(tweet), size=400))

        return self.norm_df['tweetsEmbedding']
