from PreProcessing import Preprocessing


tweetsPrp=Preprocessing()

tweetsPrp.consumer_key='Q7j4eN16sx7NWXfIysgjz4bJv'
tweetsPrp.consumer_secret='pNgJvYXIEunIPnQPHiYR3HXmCcLOgpffwYKAvHCWjeKpGHGLkI'
tweetsPrp.access_token='53767406-fgupotwM59YIC5UrxAP5yWpE4fDwqhm987T8fI2XP'
tweetsPrp.access_token_secret='5ijj5OrVDvuIpUIpSIuU9fzSgsjJX6DVwJouY9OTSkKzY'

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
