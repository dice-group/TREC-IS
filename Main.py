from Preprocessing import Preprocessing
from secrets import consumer_key, consumer_secret, access_token, access_token_secret

tweetsPrp=Preprocessing()

tweetsPrp.consumer_key=consumer_key
tweetsPrp.consumer_secret=consumer_secret
tweetsPrp.access_token=access_token
tweetsPrp.access_token_secret=access_token_secret

tweetsPrp.path='data/TRECIS-CTIT-H-Training.json'


# print event info. (id, name, desc, etc.)
events=tweetsPrp.loadEvents()
for event in events:
    print (event)


# print information types
informationTypes=tweetsPrp.loadInformationType('./data/ITR-H.types.v2.json')

for typ in informationTypes:
    print (typ['id'],'==>',typ['desc'])

tweets=tweetsPrp.loadTweets()

for tweet in tweets.TREC_data:
    print (tweet)