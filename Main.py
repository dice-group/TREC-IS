from Evaluation import Evaluation
from Preprocessing import Preprocessing
from secrets import consumer_key, consumer_secret, access_token, access_token_secret
from sklearn.naive_bayes import MultinomialNB
from Feature_Extractor import FeatureExtraction
from sklearn.model_selection import train_test_split
import numpy as np

tweetsPrp = Preprocessing(trec_path='data/TRECIS-CTIT-H-Training.json', tweets_dir='data/tweets')

tweetsPrp.consumer_key=consumer_key
tweetsPrp.consumer_secret=consumer_secret
tweetsPrp.access_token=access_token
tweetsPrp.access_token_secret=access_token_secret

training_Data = tweetsPrp.load_training_data()

# test loading training data: print tweet's categories, tweet's indicator_terms and text.
count = 0
for _, tweet in training_Data.items():
    print('categories: ',  tweet.categories, 'indicatorTerms: ', tweet.indicatorTerms, 'text: ', tweet.text, 'metadata: ', tweet.metadata)
    count = count + 1
#     #print(tweet.metadata.get('entities.hashtags') if tweet.metadata.get('entities.hashtags') else 'False')
#
#     if tweet.metadata is not None and tweet.metadata.get('entities.hashtags') is not None:
#         print('metadata: ', tweet.metadata['entities.hashtags'])
#         list_of_tweets.append({'categories': tweet.categories, 'indicatorTerms': tweet.indicatorTerms, 'text': tweet.text, 'entites_hashtags': tweet.metadata['entities.hashtags']})
#     else:
#         list_of_tweets.append(
#             {'categories': tweet.categories, 'indicatorTerms': tweet.indicatorTerms, 'text': tweet.text})
# tweet_df = pd.DataFrame(list_of_tweets, columns=['categories', 'indicatorTerms', 'text', 'entity_hashtags'])
# print(tweet_df.head(2))
print(count)
input = tweetsPrp.load_input_feature_extraction()
print(input.head(2))
# test evaluating of tweets classification with random y_true and y_predicted values:
y_true = [1, 2, 4, 1, 5, 1, 1, 2, 1, 3, 3, 1, 1, 0, 5, 2]
y_pred = [1, 3, 4, 3, 5, 1, 1, 3, 1, 3, 4, 1, 1, 2, 5, 2]

eval = Evaluation(y_true, y_pred)

print('Classification overall performance: F1 score', eval.f1_score)
print('Classification accuracy: ', eval.accuracy_score)

#---------- Feature Extraction ---------

fe = FeatureExtraction()

bow = fe.bow_features_from_tweets()
#bow = fe.tfidf_from_tweets()
df = fe.create_dataframe_for_normalized_tweets()

print(df.head(5))

train_bow, val_bow, train_cat, val_cat = train_test_split(bow, df['categories'], test_size=0.30, random_state=42)

print('train_dataset shape: ', train_bow.shape, train_cat.shape)
print('val_dataset_shape: ', val_bow.shape, val_cat.shape)

clf = MultinomialNB().fit(train_bow, train_cat)
prediction = clf.predict(val_bow)
eval = Evaluation(val_cat, prediction)

print('Classification overall performance: F1 score', eval.f1_score)
print('Classification accuracy: ', eval.accuracy_score)