import os.path
import pickle
import re

import numpy as np
import spacy
from gensim.models import KeyedVectors
from nltk import TweetTokenizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

from Helper_Feature_Extractor import Helper_FeatureExtraction
from Preprocessing import Preprocessing


class FeatureExtraction:
    def __init__(self):
        self.tweetsPrp = Preprocessing()
        self.df = self.tweetsPrp.load_input_feature_extraction()
        self.nlp = spacy.load('en')
        self.hepler_fe = Helper_FeatureExtraction()

        self.norm_df = self.create_dataframe_for_normalized_tweets()

    def create_dataframe_for_normalized_tweets(self):
        self.df.dropna(subset=['text'], how='all', inplace=True)  # drop missing values
        le = preprocessing.LabelEncoder()  # replace categorical data in 'categories' with numerical value
        self.df['categories'] = le.fit_transform(self.df['categories'])
        normalized_tweets = self.hepler_fe.extract_keywords_from_tweets(self.df)
        new_col = np.asanyarray(normalized_tweets)
        self.df['norm_tweets'] = new_col
        return self.df

    def tfidf_from_tweets(self):
        self.df = self.create_dataframe_for_normalized_tweets()
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), use_idf=True)
        feature_matrix = tfidf.fit_transform(self.df['norm_tweets'])
        return feature_matrix, tfidf

    def bow_features_from_tweets(self):
        self.df = self.create_dataframe_for_normalized_tweets()
        bow = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        feature_matrix = bow.fit_transform(self.df['norm_tweets'])
        return feature_matrix

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
            lambda tweet: TextBlob(tweet).polarity)
        return self.norm_df

    def word2vec_feature_from_tweets(self, glove_input_file, embedd_dim):
        # --- loaded saved features if it's exist ? ---
        features_path = 'features/embedding_features.pkl'
        if (os.path.exists(features_path)):
            file = open(features_path, 'rb')
            return pickle.load(file)

        # --- otherwise generate embedding features ---
        word2vec = KeyedVectors.load_word2vec_format(glove_input_file, unicode_errors='ignore',
                                                     binary=False)

        # get tfidf from each word required in embedding features
        _, tfidf_scores = self.tfidf_from_tweets()
        tfidf = dict(zip(tfidf_scores.get_feature_names(), tfidf_scores.idf_))

        # ---weighted-average tweet2vec. ---
        def build_average_Word2vec(tokens, size):
            vec = np.zeros(size)
            count = 0.
            for word in tokens:
                try:
                    vec += word2vec[word] * tfidf[word]
                    count += 1.
                except KeyError:
                    continue
            if count != 0:
                vec /= count
            return vec

        tokenizer = TweetTokenizer()
        embedd_table = {}
        for _, row in self.norm_df.iterrows():
            tweet2vec = build_average_Word2vec(tokenizer.tokenize(row['norm_tweets']), size=embedd_dim)
            embedd_table[row['tweet_id']] = tweet2vec

        # ----- saving embedding features to disk --------
        file = open(features_path, 'wb')
        pickle.dump(embedd_table, file)
        file.close()

        return embedd_table

    # ----- extract embedding and sentiment features -----
    def embedding_sentiment_features(self):
        # load saved features if it's exist ?
        feature_path = 'features/embedding_sentiment.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            return pickle.load(file)

        self.sentiment_features_from_tweets()
        embedding = self.word2vec_feature_from_tweets(glove_input_file='embeddings/glove.840B.300d.txt',
                                                      embedd_dim=300)
        for _, row in self.norm_df.iterrows():
            embedding[row['tweet_id']] = np.append(embedding[row['tweet_id']], row['sentiment'])

        # save embedding+sentiment features into disk (type: dic (tweet_id,<tweet2vec+sentiment>)
        file = open(feature_path, 'wb')
        pickle.dump(embedding, file)
        file.close()

        return embedding  # embedding and sentiment

    # ------ Bag of concepts features ----------
    def create_bag_of_concepts(self):
        '''
        For each tweet, extracts concepts from Babelnet and creates feature vectors of dimension (300, )
        :return:
        '''
        nlp = spacy.load('en_core_web_lg')
        text_col = self.norm_df['text']
        vect_col = []

        for tweet in text_col:
            tweet = self.hepler_fe.emoji_to_text(tweet)
            tweet = self.hepler_fe.expand_contractions(tweet)
            tweet = re.sub('#', '', tweet)
            tweet = re.sub('RT', '', tweet)
            concepts = self.hepler_fe.extract_concepts_from_babelnet(tweet)

            # list comprehension to get the vectors for each word
            word_vector_list = [nlp(word).vector for word in concepts]

            # calculate the mean across each word
            average_word_vector = np.mean(word_vector_list, axis=0)
            vect_col.append(average_word_vector)

        boc_array = np.asanyarray(vect_col)
        self.norm_df['bocEmbedding'] = boc_array

        return self.norm_df['bocEmbedding']




# ------------- main() for testing the code ------------- #
'''
Test embedding features, each tweet is represented as 
(1) a matrix of (n_words , word2vec).
(2) a weighted-average word2vec of all words embedding
In this code, we consider the first representation 
'''


def main():
    fe = FeatureExtraction()
    sentiments = fe.sentiment_features_from_tweets()
    embedd_senti = fe.embedding_sentiment_features()


if __name__ == '__main__':
    main()
