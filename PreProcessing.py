import json
import pandas as pd

from twarc import Twarc
from Tweets import Tweets


class Preprocessing:

    def __init__(self, consumer_key=None, consumer_secret=None,
                 access_token=None, access_token_secret=None,path=None):

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret

        self.path=path

    def loadEvents(self):
        '''
        :param path: path of TREC-IS training data
        :return: events annotation information for all events as a DataFrame: (identifier, name, description,
        type, imageURL, annotationTableName)
        '''

        events_annotation = json.load(open(self.path))
        events = pd.DataFrame.from_dict(events_annotation['annotator'], orient='columns')['eventsAnnotated']

        return events

    def loadTweets(self):
        '''
        :param path: path of TREC-IS training data
        :return: list of tweets, each tweet has standard attributes as in
        https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.html
        '''

        events = json.load(open(self.path))
        events=pd.DataFrame.from_dict(events['events'],orient='columns')

        tweetsIDs = []

        for _,event in events.iterrows():
            for tweet in event['tweets']:
                tweetsIDs.append(tweet['postID'])

        t = Twarc(self.consumer_key, self.consumer_secret, self.access_token, self.access_token_secret)

        tweets_fullData = t.hydrate(iter(tweetsIDs))  # access all tweets

        return Tweets(event['tweets'],tweets_fullData)

    def loadInformationType(self, path):
        '''
        :param path: file path of information type
        :return: information types data: (id, desc, level, intentType, exampleLowLevelTypes)
        '''
        return pd.read_json(path, orient='columns')['informationTypes']
