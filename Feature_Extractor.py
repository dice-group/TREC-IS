import itertools
import os.path
import pickle
import re

import numpy as np
import pandas as pd

pd.set_option('mode.chained_assignment', None)

import spacy

from gensim.models import KeyedVectors
from nltk import TweetTokenizer

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from textblob import TextBlob
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import MultiLabelBinarizer

from Helper_Feature_Extractor import Helper_FeatureExtraction

from Preprocessing import Preprocessing
from DeepModel import Model
from Evaluate_Models import ModelEvaluation

from FeaturePyramids import Features

class FeatureExtraction:
    def __init__(self):
        self.tweetsPrp = Preprocessing()
        self.df = self.tweetsPrp.load_input_feature_extraction()
        self.nlp = spacy.load('en')
        self.hepler_fe = Helper_FeatureExtraction()
        self.norm_df = self.create_dataframe_for_normalized_tweets()
        self.tfidf_feature, self.tfidf = self.tfidf_from_tweets()
        self.countVec_feature = self.countVec_from_tweets()

    def reduce_dimensions(self, feature_matrix, n_components=300, method='pca'):
        '''

        :param feature_matrix:
        :param n_components:
        :param method: 'pca', 'svd'
        :return:
        '''
        if method=='pca':
            pca = PCA(n_components=n_components)
            matrix_reduced = pca.fit_transform(feature_matrix)
            return matrix_reduced

        elif method=='svd':
            svd = TruncatedSVD(n_components)
            matrix_reduced = svd.fit_transform(feature_matrix)
            return matrix_reduced


    def create_dataframe_for_normalized_tweets(self):
        self.df.dropna(subset=['text'], how='all', inplace=True)  # drop missing values
        le = preprocessing.LabelEncoder()  # replace categorical data in 'categories' with numerical value
        self.df['categories'] = le.fit_transform(self.df['categories'])
        normalized_tweets = self.hepler_fe.extract_keywords_from_tweets(self.df)
        new_col = np.asanyarray(normalized_tweets)
        self.df['norm_tweets'] = new_col
        return self.df

    def tfidf_from_tweets(self, dimensionality_reduction=False, method='pca', n_components=300, analyzer='word', norm='l2', ngram_range=(1, 1), use_idf=True,
                          preprocessor=None, tokenizer=None, stop_words=None, max_df=1.0, min_df=1,
                          max_features=None, vocabulary=None, smooth_idf=True, sublinear_tf=False):

        tfidf = TfidfVectorizer(analyzer= analyzer, norm= norm, ngram_range= ngram_range, use_idf= use_idf,
                    preprocessor= preprocessor, tokenizer= tokenizer, stop_words= stop_words,
                    max_df= max_df, min_df= min_df,  max_features= max_features, vocabulary= vocabulary,
                    smooth_idf= smooth_idf, sublinear_tf= sublinear_tf)

        feature_matrix = tfidf.fit_transform(self.norm_df['norm_tweets'])

        if dimensionality_reduction:
            return self.reduce_dimensions(feature_matrix.toarray(), n_components=n_components, method=method), tfidf

        return feature_matrix.toarray(), tfidf

    def countVec_from_tweets(self, dimensionality_reduction=False, method='pca', n_components=300, analyzer='word', ngram_range=(1, 1),
                             preprocessor=None, tokenizer=None, stop_words=None,
                     max_df=1.0, min_df=1,  max_features=None, vocabulary=None ):

        count_vec = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words,
                     max_df=max_df, min_df=min_df,  max_features=max_features, vocabulary=vocabulary )

        feature_matrix = count_vec.fit_transform(self.norm_df['norm_tweets'])

        if dimensionality_reduction:
            return self.reduce_dimensions(feature_matrix.toarray(), n_components=n_components, method=method)

        return feature_matrix.toarray()

    def bow_features(self, mode='countVec', norm='l2', dimensionality_reduction=False, method='pca', n_components=300, analyzer='word', ngram_range=(1, 1),
                     use_idf=True, preprocessor=None, tokenizer=None, stop_words=None,
                     max_df=1.0, min_df=1,  max_features=None, vocabulary=None, smooth_idf=True, sublinear_tf=False):
        '''

        :param mode: {'countVec', 'tfidf'}
        :param norm: used to normalize term vectors {'l1', 'l2', None}
        :param dimensionality_reduction: {'true', 'false'}
        :param method: {'pca', 'svd'}
        :param n_components: int, reduced dimesion = 300 by default
        :param analyzer: {'word', 'char'} or callable for tf-idf , {‘word’, ‘char’, ‘char_wb’} or callable for countVec
        :param ngram_range: tuple(min_n, max_n)
        :param use_idf: boolean, default = True
        :param preprocessor: callable or None (default)
        :param tokenizer: callable or None (default)
        :param stop_words: string {‘english’}, list, or None (default)
        :param max_df: float in range [0.0, 1.0] or int, default=1.0
        :param min_df: float in range [0.0, 1.0] or int, default=1
        :param max_features: int or None, default=None
        :param vocabulary: Mapping or iterable, optional
        :param smooth_idf: boolean, default=True
        :param sublinear_tf: boolean, default=False
        :return:
        '''
        # --- loaded saved features if it's exist ? ---
        features_path = 'features/bow.pkl'
        if (os.path.exists(features_path)):
            file = open(features_path, 'rb')
            return pickle.load(file)

        if mode == 'countVec':
            feature_matrix = self.countVec_from_tweets(dimensionality_reduction=dimensionality_reduction,method=method, n_components=n_components, analyzer=analyzer,
                                                ngram_range=ngram_range, preprocessor=preprocessor, tokenizer=tokenizer,
                                                stop_words=stop_words, max_df=max_df, min_df=min_df,
                                                max_features=max_features, vocabulary=vocabulary)
        else:
            # tf-idf - returns feature_matrix, tfidf mapping
            feature_matrix, tfidf = self.tfidf_from_tweets(dimensionality_reduction=dimensionality_reduction, method=method, n_components=n_components, analyzer= analyzer,
                                             norm= norm, ngram_range= ngram_range, use_idf= use_idf,
                                             preprocessor= preprocessor, tokenizer= tokenizer, stop_words= stop_words,
                                             max_df= max_df, min_df= min_df,  max_features= max_features, vocabulary= vocabulary,
                                             smooth_idf= smooth_idf, sublinear_tf= sublinear_tf)

        embedd_table = {}
        for row, feature_vec in zip(self.norm_df['tweet_id'], feature_matrix):
            embedd_table[row] = feature_vec

        # ----- saving embedding features to disk --------
        file = open(features_path, 'wb')
        pickle.dump(embedd_table, file)
        file.close()

        return embedd_table


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

        feature_path = 'features/boc_wordEmbeddings.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            return pickle.load(file)

        nlp = spacy.load('en_core_web_lg')
        text_col = self.norm_df['text']
        vect_col = []
        count = 0
        zero_list = []

        for tweet in text_col:
            tweet = self.hepler_fe.emoji_to_text(tweet)
            tweet = self.hepler_fe.expand_contractions(tweet)
            tweet = re.sub('#', '', tweet)
            tweet = re.sub('RT', '', tweet)
            concepts = self.hepler_fe.extract_concepts_from_babelnet(tweet)

            # list comprehension to get the vectors for each word
            word_vector_list = [nlp(word).vector for word in concepts]
            # print(len(word_vector_list), ' : ', word_vector_list)

            # calculate the mean/sum across each word
            average_word_vector = np.sum(word_vector_list, axis=0, keepdims=False)
            # print('avg word vector : ' , average_word_vector.shape, average_word_vector)
            if(np.sum(word_vector_list) == 0):
                zero_list.append((tweet, average_word_vector))
                count += 1
            vect_col.append(average_word_vector)

        print(count)
        print(zero_list)
        boc_array = np.asanyarray(vect_col)
        self.norm_df['bocEmbedding'] = boc_array

        embed_table = {}

        for row, feature in zip(self.norm_df['tweet_id'], self.norm_df['bocEmbedding']):
            embed_table[row] = feature

        # print(embed_table)

        # save embedding features into disk
        file = open(feature_path, 'wb')
        pickle.dump(embed_table, file)
        file.close()

        return embed_table

    def encode_synsets_from_babelfy(self):
        '''
        Uses one-hot encoding to create feature_vectors from the synsets returned by Babelfy
        :param:
        :return:
        '''

        feature_path = 'features/boc_OHE.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            return pickle.load(file)

        text_col = self.norm_df['text']
        all_synsets = []   # a list of all synsets in the dataset
        tweet_synsets = [] #for each tweet, preserves its synsets

        for tweet in text_col:
            synset_list = []   # list of synsetIDs for one tweet
            tweet = self.hepler_fe.emoji_to_text(tweet)
            tweet = self.hepler_fe.expand_contractions(tweet)
            tweet = re.sub('#', '', tweet)
            tweet = re.sub('RT', '', tweet)
            synsetDict, scoreDict = self.hepler_fe.extract_synsets_from_babelfy(tweet)
            for key in synsetDict.keys():
                all_synsets.append(synsetDict[key])
                synset_list.append(synsetDict[key])
            tweet_synsets.append(synset_list)

        print('all tweet synsets: ', tweet_synsets)
        iterable_tweet_synsets = itertools.chain.from_iterable(tweet_synsets)

        # create a dictionary that maps synsets to numerical ids
        synset_to_id = {token: idx+1 for idx, token in enumerate(set(iterable_tweet_synsets))}

        print('synset_to_ids: ', synset_to_id)

        # convert synset lists to id-lists
        synset_ids =[[synset_to_id[token] for token in synset_list] for synset_list in tweet_synsets]
        print('synset_ids: ', synset_ids)
        print('total synsets: ', len(synset_to_id))

        # convert list of synset_ids to one-hot representation
        mlb = MultiLabelBinarizer()
        boc_features = mlb.fit_transform(tweet_synsets)

        embed_table = {}

        for row, feature in zip(self.norm_df['tweet_id'], boc_features):
            embed_table[row] = feature

        # save one-hot vectors into disk (type: {tweetID : <one-hot vector>} )
        file = open(feature_path, 'wb')
        pickle.dump(embed_table, file)
        file.close()

        return embed_table

    # ----- extract bow and boc features -----
    def bow_boc_features(self, mode='countVec', norm='l2', dimensionality_reduction=True,n_components=300, analyzer='word', ngram_range=(1, 1),
                     use_idf=True, preprocessor=None, tokenizer=None, stop_words=None,
                     max_df=0.98, min_df=1,  max_features=None, vocabulary=None, smooth_idf=True, sublinear_tf=False):

        # load saved features if it's exist ?
        feature_path = 'features/boc-bow.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            return pickle.load(file)

        boc_table = self.encode_synsets_from_babelfy()   # --- one-hot encoders ----

        #boc_table = self.create_bag_of_concepts()

        bow_table = self.bow_features(mode=mode, norm=norm, dimensionality_reduction=dimensionality_reduction,n_components=n_components, analyzer=analyzer, ngram_range=ngram_range,
                     use_idf=use_idf, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words,
                     max_df=max_df, min_df=min_df,  max_features=max_features, vocabulary=vocabulary, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)

        for _, row in self.norm_df.iterrows():
            bow_table[row['tweet_id']] = np.append(bow_table[row['tweet_id']], boc_table[row['tweet_id']])


        for row in bow_table:
            print(row)
            #bow_table[row] = np.append(bow_table[row], boc_table[row])

        # save bow + boc features into disk (type: dic (tweet_id,<bow + boc>)
        file = open(feature_path, 'wb')
        pickle.dump(bow_table, file)
        file.close()

        return bow_table  # bow and boc

    # ----- extract bow and sentiment features -----
    def bow_sentiment_features(self):
        # load saved features if it's exist ?
        feature_path = 'features/bow_sentiment.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            return pickle.load(file)

        self.sentiment_features_from_tweets()
        bow_dict = self.bow_features(mode='countVec', norm='l2', dimensionality_reduction=True, method= 'svd', n_components=300, analyzer='word', ngram_range=(1, 1),
                     use_idf=True, preprocessor=None, tokenizer=None, stop_words=None,
                     max_df=0.98, min_df=1,  max_features=None, vocabulary=None, smooth_idf=True, sublinear_tf=False)

        for _, row in self.norm_df.iterrows():
            bow_dict[row['tweet_id']] = np.append(bow_dict[row['tweet_id']], row['sentiment'])

        file = open(feature_path, 'wb')
        pickle.dump(bow_dict, file)
        file.close()

        return bow_dict  # bow and sentiment

    # ----- extract boc and sentiment features -----
    def boc_sentiment_features(self):
        # load saved features if it's exist ?
        feature_path = 'features/boc_sentiment.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            return pickle.load(file)

        self.sentiment_features_from_tweets()
        boc_dict = self.encode_synsets_from_babelfy()

        for _, row in self.norm_df.iterrows():
            boc_dict[row['tweet_id']] = np.append(boc_dict[row['tweet_id']], row['sentiment'])

        file = open(feature_path, 'wb')
        pickle.dump(boc_dict, file)
        file.close()

        return boc_dict  # boc and sentiment

    def sentiment_features(self):
        feature_path = 'features/sentiment.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            return pickle.load(file)

        self.sentiment_features_from_tweets()

        sent_dict = {}

        for _, row in self.norm_df.iterrows():
            sent_dict[row['tweet_id']] = [row['sentiment']]

        file = open(feature_path, 'wb')
        pickle.dump(sent_dict, file)
        file.close()

        return sent_dict  # sentiment


# ------------- main() for testing the code ------------- #
'''
Test embedding features, each tweet is represented as 
(1) a matrix of (n_words , word2vec).
(2) a weighted-average word2vec of all words embedding
In this code, we consider the first representation 
'''

def main():
    fe = FeatureExtraction()

    print(fe.sentiment_features())
    # print(fe.boc_sentiment_features())
    # print(fe.bow_sentiment_features())

    feat_pyramids = Features()

    # --- load training data ---
    data = fe.norm_df[['tweet_id', 'categories']]
    data.set_index('tweet_id', inplace=True)

    embedding_dict, bow_dict, boc_dict, sent_dict, bow_sent, boc_sent, embedding_sent_dict, \
    embedding_sent_bow, embedding_sent_boc, bow_boc, embedding_bow, embedding_boc, bow_sent_boc, \
    bow_boc_embedding, embedding_sent_bow_boc = feat_pyramids.get_all_features()

    feature_list = [embedding_dict, bow_dict, boc_dict, sent_dict, bow_sent, boc_sent, embedding_sent_dict,
               embedding_sent_bow, embedding_sent_boc, bow_boc, embedding_bow, embedding_boc, bow_sent_boc,
               bow_boc_embedding, embedding_sent_bow_boc]

    i=0

    for feature in feature_list:

        data['feature_set'+str(i)] = np.nan
        data['feature_set'+str(i)] = data['feature_set'+str(i)].astype(object)

        for id, row in data.iterrows():
            data.at[id, 'feature_set'+str(i)] = feature[id]

        # print(data['feature_set'+str(i)][:4])

        #---- evaluation ----
        print(type(data['feature_set'+str(i)].tolist()))
        modelEval =  ModelEvaluation(X=data['feature_set'+str(i)].tolist(), y=data['categories'].tolist(), feature_name='feature_set'+str(i))
        modelEval.run_evaluation()
        i += 1


    # # -- Ok let's train a simple deep model --
    # simpleModel = Model(X=data['emb_senti_features'].tolist(), y=data['categories'].tolist())
    # simpleModel.simple_DeepModel()
    #
    # simpleModel.evaluate_model()

if __name__ == '__main__':
    main()

