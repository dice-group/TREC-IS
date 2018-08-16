import csv
import json
import os
import pickle
from glob import glob

import pandas as pd
from twarc import Twarc

from Tweet import Tweet


class Preprocessing:
    def __init__(self, consumer_key=None, consumer_secret=None,
                 access_token=None, access_token_secret=None, trec_path=None, tweets_dir=None):
        '''
        :param consumer_key, consumer_secret, access_token, access_token_secret: Twitter authentication tokens
        :param trec_path: path of training data (json file)
        :param tweets_dir:path of tweets directory
        '''
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret

        self.trec_path = trec_path
        self.tweets_dir = tweets_dir

    def load_Events(self, path):
        '''
        load events information from the json file.
        :return: all events information as a DataFrame: (identifier, name, description,type, imageURL, annotationTableName)
        '''
        events_annotation = json.load(open(path))
        events = pd.DataFrame.from_dict(events_annotation['annotator'], orient='columns')['eventsAnnotated']

        return events

    def load_InformationType(self, path):
        '''
        loads information about tweet's categories (information types)
        :param path: path of information_type file.
        :return: information_type's data: (id, desc, level, intentType, exampleLowLevelTypes)
        '''
        return pd.read_json(path, orient='columns')['informationTypes']

    def load_Tweets(self):
        '''
        load and combine tweets (downloaded by TREC-Downloader) from json files into a dictionary.
        :return: a dictionary of all tweets {id: Tweet} , Tweet object has info as (id, full_text and metadata)
        '''
        all_tweets = {}

        # loading all json files
        for f_path in glob(self.tweets_dir + '/*.json'):
            try:
                tweets_json = json.load(open(f_path))
            except:
                print(f_path)

            f_name = os.path.basename(f_path)[
                     :-5]  # get only json file_name and used as an identifier for event tweets.
            for tweet in tweets_json[f_name]:
                all_tweets[tweet['identifier']] = Tweet(tweet['identifier'], tweet['text'], tweet['metadata'])

        return all_tweets

    def get_traing_data(self):
        '''
        :return: combined data (tweets info and trec-is data) as dictionary {tweet_id: Tweet}
        '''
        # load tweets retrieved by TREC-Tweets downloader
        retrieved_tweets = self.load_Tweets()

        missed_tweets = []
        training_data = {}  # dict {'tweet id': Tweet}

        # load TREC data data: tweetsID, tweet_priority, tweet_categories, indicator_terms
        events = json.load(open(self.trec_path))
        events = pd.DataFrame.from_dict(events['events'], orient='columns')

        for _, event in events.iterrows():
            for trec_tweet in event['tweets']:
                if trec_tweet['postID'] in retrieved_tweets:  # check if tweets_full is retrieved ?
                    retriev_tweet = retrieved_tweets[trec_tweet['postID']]
                    training_data[trec_tweet['postID']] = Tweet(id=retriev_tweet.id, text=retriev_tweet.text,
                                                                metadata=retriev_tweet.metadata,
                                                                priority=trec_tweet['priority'],
                                                                indicatorTerms=trec_tweet['indicatorTerms'],
                                                                categories=trec_tweet['categories'])
                else:
                    # adding missed tweets
                    training_data[trec_tweet['postID']] = Tweet(id=trec_tweet['postID'],
                                                                priority=trec_tweet['priority'],
                                                                indicatorTerms=trec_tweet['indicatorTerms'],
                                                                categories=trec_tweet['categories'])
                    missed_tweets.append(trec_tweet['postID'])

        # Retrieve the missed tweets by Twarc tool and combine with training data
        t = Twarc(self.consumer_key, self.consumer_secret, self.access_token, self.access_token_secret)

        tweets_twarc = t.hydrate(iter(missed_tweets))  # retrieve all tweets by IDs
        for twtt in tweets_twarc:
            training_data[str(twtt['id'])].add_tweets_data(twtt['full_text'], twtt['entities'])

        return training_data

    def save_trainingData(self):
        trainingData = self.get_traing_data()
        # save TREC-data as an object
        trec_data_path = 'data/TREC-data.pkl'
        file = open(trec_data_path, 'wb')
        pickle.dump(trainingData, file)
        file.close()

        # save TREC-data as csv
        with open('data/TREC-data.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['tweet_id', 'categories', 'text', 'indicatorTerms', 'metadata', 'priority'])
            for key, value in trainingData.items():
                tweet = trainingData[key]
                writer.writerow([tweet.id, tweet.categories, tweet.text, tweet.indicatorTerms,
                                 tweet.metadata, tweet.priority])
        csv_file.close()

    def load_training_data(self, trec_data_path='data/TREC-data.pkl'):
        if (os.path.exists(trec_data_path)):
            file = open('data/preprocessed_data.pkl', 'rb')
            trainingData = pickle.load(file)
            file.close()

        return trainingData

    def load_input_feature_extraction(self):
        data = self.load_training_data()
        tweet_list = []
        for _, tweet in data.items():
            # if tweet.metadata is not None and tweet.metadata.get('entities.hashtags') is not None:
            tweet_list.append(
                {'tweet_id': tweet.id, 'categories': tweet.categories, 'indicatorTerms': tweet.indicatorTerms,
                 'text': tweet.text, 'metadata': tweet.metadata})
            # else:
            #     tweet_list.append(
            #         {'categories': tweet.categories, 'indicatorTerms': tweet.indicatorTerms, 'text': tweet.text})
        tweet_df = pd.DataFrame(tweet_list, columns=['tweet_id', 'categories', 'indicatorTerms', 'text', 'metadata'])

        return tweet_df


# --- main() for testing the code ---
def main():
    from secrets import consumer_key, consumer_secret, access_token, access_token_secret

    tweetsPrp = Preprocessing(trec_path='data/TRECIS-CTIT-H-Training.json', tweets_dir='data/tweets')

    tweetsPrp.consumer_key = consumer_key
    tweetsPrp.consumer_secret = consumer_secret
    tweetsPrp.access_token = access_token
    tweetsPrp.access_token_secret = access_token_secret

    tweetsPrp.save_trainingData()

    # -- load Information types --
    #
    #  tweetsPrp = Preprocessing()
    #
    # # loading information types in test data
    # information_types = tweetsPrp.load_InformationType('data/test data/ITR-H.types.v2.json')
    #
    # for _, item in information_types.iteritems():
    #     print(item['id'], ' => ', item['desc'])

if __name__ == '__main__':
    main()
