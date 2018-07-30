from Evaluation import Evaluation
from Preprocessing import Preprocessing
from secrets import consumer_key, consumer_secret, access_token, access_token_secret

tweetsPrp = Preprocessing(trec_path='data/TRECIS-CTIT-H-Training.json', tweets_dir='data/tweets')

tweetsPrp.consumer_key=consumer_key
tweetsPrp.consumer_secret=consumer_secret
tweetsPrp.access_token=access_token
tweetsPrp.access_token_secret=access_token_secret

training_Data = tweetsPrp.load_training_data()
# test loading training data: print tweet's categories, tweet's indicator_terms and text.
for _, tweet in training_Data.items():
    print(tweet.categories, tweet.indicatorTerms, tweet.text)

# test evaluating of tweets classification with random y_true and y_predicted values:
y_true = [1, 2, 4, 1, 5, 1, 1, 2, 1, 3, 3, 1, 1, 0, 5, 2]
y_pred = [1, 3, 4, 3, 5, 1, 1, 3, 1, 3, 4, 1, 1, 2, 5, 2]

eval = Evaluation(y_true, y_pred)

print('Classification overall performance: F1 score', eval.f1_score)
print('Classification accuracy: ', eval.accuracy_score)
