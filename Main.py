import pandas as pd
import spacy
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from Evaluation import Evaluation
from Feature_Extractor import FeatureExtraction
from Helper_Feature_Extractor import Helper_FeatureExtraction
from Preprocessing import Preprocessing
from secrets import consumer_key, consumer_secret, access_token, access_token_secret

tweetsPrp = Preprocessing(trec_path='data/TRECIS-CTIT-H-Training.json', tweets_dir='data/tweets')

tweetsPrp.consumer_key=consumer_key
tweetsPrp.consumer_secret=consumer_secret
tweetsPrp.access_token=access_token
tweetsPrp.access_token_secret=access_token_secret

training_Data = tweetsPrp.load_training_data()

# test loading training data: print tweet's categories, tweet's indicator_terms and text.
count = 0
for _, tweet in training_Data.items():
    #print('categories: ',  tweet.categories, 'indicatorTerms: ', tweet.indicatorTerms, 'text: ', tweet.text, 'metadata: ', tweet.metadata)
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
#print(count)
input = tweetsPrp.load_input_feature_extraction()
#print(input.head(2))
# test evaluating of tweets classification with random y_true and y_predicted values:
y_true = [1, 2, 4, 1, 5, 1, 1, 2, 1, 3, 3, 1, 1, 0, 5, 2]
y_pred = [1, 3, 4, 3, 5, 1, 1, 3, 1, 3, 4, 1, 1, 2, 5, 2]

eval = Evaluation(y_true, y_pred)

#print('Classification overall performance: F1 score', eval.f1_score)
#print('Classification accuracy: ', eval.accuracy_score)

##check normalized tweet
helper = Helper_FeatureExtraction()
nlp = spacy.load('en')
text = "the no. 1 tourist spot in cagayan de oro 👐🙌👈 #bridge #rotonda #flood #highflood #omg #pabloph # @ the bridge http://t.co/upvoomhi"
print(helper.normalize_tweet(text=text, nlp=nlp, lemmatization= False, ))

#---------- Feature Extraction ---------

fe = FeatureExtraction()

bow = fe.bow_feature
# bow = fe.tfidf_feature

df = fe.get_dataframe_for_normalized_tweets()

print(df.head(5))

models = [
    RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42),
    LinearSVC(random_state=42),
    MultinomialNB(),
    LogisticRegression(random_state=42, solver='newton-cg'),
    tree.DecisionTreeClassifier(random_state=42)

]

kf = 10

entries = []
for model in models:
    model_name = model.__class__.__name__
    scores = cross_val_score(model, bow, df['categories'], scoring='accuracy', cv=kf)
    for fold_idx, accuracy in enumerate(scores):
        #print (model_name, fold_idx, accuracy)
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

#-----------visualization of models and accuracies---------
# sns.boxplot(x='model_name', y = 'accuracy', data=cv_df)
# plt.show()

mean_acc = cv_df.groupby('model_name').accuracy.mean()
print ('average accuracy: ', mean_acc)

#clf = MultinomialNB().fit(train_bow, train_cat)
#clf = MultinomialNB()

#scores = cross_validate(clf, train_bow, train_cat, scoring=scoring, cv=10, return_train_score=True)
#sorted(scores.keys())
#print(scores)
#prediction = clf.predict(val_bow)
#eval = Evaluation(val_cat, prediction)


#print('Classification overall performance: F1 score', eval.f1_score)
#print('Classification accuracy: ', eval.accuracy_score)

# ------------------ Testing Sentiment and Embedding Features---------#
'''
Hint: 
- sentiment analysis is performed in tweet's full text without normalization to keep stop words which preserve tweet's meaning. 
- tweets Embedding feature is a weighted average of word2vec vectors in a tweet. word2vec is computed by a pre-trained word 
  embedding model on a dataset of tweets 
'''
fe.sentiment_features_from_tweets()
fe.word2vec_feature_from_tweets()

tweets_sentiments_embedding = fe.norm_df[['text', 'sentiment', 'tweetsEmbedding']]
print(tweets_sentiments_embedding)
# ------------------------------------------------------
