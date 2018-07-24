from Preprocessing import Preprocessing
from secrets import consumer_key, consumer_secret, access_token, access_token_secret

tweetsPrp = Preprocessing(trec_path='data/TRECIS-CTIT-H-Training.json', tweets_dir='data/tweets')

tweetsPrp.consumer_key=consumer_key
tweetsPrp.consumer_secret=consumer_secret
tweetsPrp.access_token=access_token
tweetsPrp.access_token_secret=access_token_secret

training_Data = tweetsPrp.load_traing_data()

# test loading training data: print tweet's categories, tweet's indicator_terms and text.
for _, tweet in training_Data.items():
    print(tweet.categories, tweet.indicatorTerms, tweet.text)
