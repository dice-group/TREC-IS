
from sklearn.model_selection import train_test_split


from tpot import TPOTClassifier
from Feature_Extractor import FeatureExtraction


fe = FeatureExtraction()

df = fe.norm_df
print(df.head(5))

tpot = TPOTClassifier(periodic_checkpoint_folder='tpot_progress',warm_start=True, verbosity=2,
                      early_stop=5000, random_state=42, n_jobs= -1, cv=2)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['categories'],
                                                    train_size=0.75, test_size=0.25)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_pipeline.py')

X_train, X_test, y_train, y_test = train_test_split(df['norm_tweets'], df['categories'],
                                                    train_size=0.75, test_size=0.25)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_norm_pipeline.py')

