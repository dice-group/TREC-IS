import pandas as pd
import spacy, collections
from collections import Counter
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import pickle

from tpot import TPOTClassifier
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

input = tweetsPrp.load_input_feature_extraction()

# test evaluating of tweets classification with random y_true and y_predicted values:
y_true = [1, 2, 4, 1, 5, 1, 1, 2, 1, 3, 3, 1, 1, 0, 5, 2]
y_pred = [1, 3, 4, 3, 5, 1, 1, 3, 1, 3, 4, 1, 1, 2, 5, 2]

eval = Evaluation(y_true, y_pred)

#print('Classification overall performance: F1 score', eval.f1_score)
#print('Classification accuracy: ', eval.accuracy_score)

# ------------check normalized tweet--------------------------

helper = Helper_FeatureExtraction()
nlp = spacy.load('en')
text = "the no. 1 tourist spot in cagayan de oro 👐🙌👈 #bridge #rotonda #flood #highflood #omg #pabloph # @ the bridge http://t.co/upvoomhi"
print(helper.normalize_tweet(text=text, nlp=nlp, lemmatization= False, ))

# for _, tweet in training_Data.items():
#     if tweet.text:
#         helper.expand_twitterLingo(tweet.text)

# helper = Helper_FeatureExtraction()
# nlp = spacy.load('en')
# text = "the no. 1 tourist spot in cagayan de oro 👐🙌👈 #bridge #rotonda #flood #highflood #omg #pabloph # @ the bridge http://t.co/upvoomhi"
# #text = """rt @itsshowtime: #pabloph
# ▬▬▬▬▬▬▬▬▬▬▬▬▬ஜ۩۞۩ஜ▬▬▬▬▬▬▬▬▬▬▬▬▬▬
# ✞✞✞ ✞ ｐｒａｙ ｆｏｒ ｔｈｅ ｐｈｉｌｉｐｐｉｎｅｓ ✞✞✞✞
# ▬▬▬▬▬▬▬▬▬▬▬▬▬ஜ۩۞۩ஜ▬▬▬▬▬▬▬▬▬▬▬▬▬▬"""
# print('emoji to text: ', helper.emoji_to_text(text))
# print('no emojis: ', helper.remove_emojis(text))
# print(helper.normalize_tweet(text=text, nlp=nlp, lemmatization= False, ))
# text = helper.emoji_to_text(text)
# print('concepts: ', helper.extract_concepts_from_babelnet(text))


#---------- Feature Extraction ---------

fe = FeatureExtraction()


bow_dict = fe.bow_features(mode='countVec', norm='l2', dimensionality_reduction=True, method='svd', n_components=300,
                           analyzer='word', ngram_range=(1, 1), use_idf=True, preprocessor=None, tokenizer=None,
                           stop_words=None, max_df=0.98, min_df=1, max_features=None, vocabulary=None, smooth_idf=True,
                           sublinear_tf=False)


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


data = fe.norm_df[['tweet_id', 'categories']]
data.set_index('tweet_id', inplace=True)
data['bow_features'] = np.nan
data['bow_features'] = data['bow_features'].astype(object)


for id, row in data.iterrows():
    data.at[id, 'bow_features'] = bow_dict[id]

df = fe.norm_df
print(df.head(5))

models = [
    RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42),
    LinearSVC(random_state=42),
    #MultinomialNB(),
    LogisticRegression(random_state=42, solver='newton-cg'),
    tree.DecisionTreeClassifier(random_state=42)

]

kf = 2

entries = []
for model in models:
    model_name = model.__class__.__name__
    scores = cross_val_score(model, data['bow_features'].tolist(), data['categories'].tolist() , scoring='accuracy', cv=kf)
    for fold_idx, accuracy in enumerate(scores):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# -----------visualization of models and accuracies---------
# sns.boxplot(x='model_name', y = 'accuracy', data=cv_df)
# plt.show()

mean_acc = cv_df.groupby('model_name').accuracy.mean()
print ('average accuracy: ', mean_acc)

#------------------- Bag of Concepts ---------------------------------#
'''
Concepts are extracted from BabelNet (and Babelfy) after replacing emojis with text, expanding contractions and removing '#' and 'RT'
from each tweet.   
'''

data = fe.norm_df[['tweet_id', 'categories']]
data.set_index('tweet_id', inplace=True)
data['boc_features'] = np.nan
data['boc_features'] = data['boc_features'].astype(object)

# boc_features = fe.encode_synsets_from_babelfy()
boc_features = fe.create_bag_of_concepts()
print(fe.norm_df.head(5))

for id, row in data.iterrows():
    data.at[id, 'boc_features'] = boc_features[id]

entries = []
for model in models:
    model_name = model.__class__.__name__
    scores = cross_val_score(model, data['boc_features'].tolist(), data['categories'].tolist(), scoring='accuracy', cv=kf)
    for fold_idx, accuracy in enumerate(scores):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print('Accuracy with Bag of concepts')
mean_acc = cv_df.groupby('model_name').accuracy.mean()
print ('average accuracy: ', mean_acc)


#------------------- Testing BOW and BOC-encoding features -----------#

# embed_dict = fe.bow_boc_features()
data['bow_boc'] = np.nan
data['bow_boc'] = data['bow_boc'].astype(object)

bow_boc = pickle.load(open('features/boc-bow.pkl', 'rb'))

for id, row in data.iterrows():
    data.at[id, 'bow_boc'] = bow_boc[id]

entries = []
for model in models:
    model_name = model.__class__.__name__
    scores = cross_val_score(model, data['bow_boc'].tolist(), data['categories'].tolist(), scoring='accuracy', cv=kf)
    for fold_idx, accuracy in enumerate(scores):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
print('Accuracy with BOW-BOC')
mean_acc = cv_df.groupby('model_name').accuracy.mean()
print ('average accuracy: ', mean_acc)



# ------------------ Testing Sentiment and Embedding Features---------#
'''
Hint: 
- sentiment analysis is performed in tweet's full text without normalization to keep stop words which preserve tweet's meaning. 
- tweets Embedding feature is a weighted average of word2vec vectors in a tweet. word2vec is computed by a pre-trained word 
  embedding model on a dataset of tweets 
'''
fe.sentiment_features_from_tweets()
fe.word2vec_feature_from_tweets()

tweets_sentiments_embedding = fe.norm_df[['sentiment', 'tweetsEmbedding']]
print(tweets_sentiments_embedding)
# ------------------------------------------------------
