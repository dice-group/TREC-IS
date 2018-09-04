import itertools as it
import pickle
import numpy as np
import os

class Features:

    def feature_permutation(self, features):
        '''
        :param features: list of features extracted from tweets
        :return: a list of all possible combination of features
        '''
        all_the_features = []
        for r in range(1, len(features) + 1):
            all_the_features = all_the_features + list(it.combinations(features, r))

        feature_comb = []
        for feature in all_the_features:

            if (len(feature) < 2):
                continue

            feature_set = []
            for t in feature:
                feature_set += list(t)

            feature_comb.append(feature_set)

        return feature_comb

    def features_pyramids(self, features_list):
        featurePyramids = []

        for i in range(len(features_list[0])):  # range of dataset size (i: row 0 to len(data))
            feature_row = []

            for feature in features_list:
                feature_row.append(feature[i])

            # print (feature_row)

            feature_comb = self.feature_permutation(feature_row)
            featurePyramids.append(feature_comb)

        f_count = len(featurePyramids[0])
        all_features = {k: [] for k in range(f_count)}

        for feature in featurePyramids:

            for i in range(f_count):
                if i in all_features:
                    all_features[i].append(feature[i])
                else:
                    all_features[i] = feature[i]

        return all_features

    def get_all_features(self, name='default'):

        # loading saved features
        datetime_dict = pickle.load(open('features/datetime-'+ name +'.pkl', 'rb'))
        sent_dict = pickle.load(open('features/sentiment-'+ name +'.pkl', 'rb'))
        bow_sent =  pickle.load(open('features/bow_sentiment-'+ name +'.pkl', 'rb'))
        boc_sent = pickle.load(open('features/boc_sentiment-'+ name +'.pkl', 'rb'))
        embedding_dict = pickle.load(open('features/embedding_features-'+ name +'.pkl', 'rb'))
        embedding_sent_dict = pickle.load(open('features/embedding_sentiment-'+ name +'.pkl', 'rb'))
        bow_dict = pickle.load(open('features/bow-'+ name +'.pkl', 'rb'))
        boc_dict = pickle.load(open('features/boc_OHE-'+ name +'.pkl', 'rb'))
        # event_dict = pickle.load(open('features/boc_OHE-'+ name +'.pkl', 'rb'))

        # embedding_feat = []
        # bow_feat = []
        # boc_feat = []

        embedding_bow = {}
        embedding_boc = {}
        bow_sent_boc = {}
        bow_boc_embedding = {}
        embedding_sent_bow = {}
        embedding_sent_boc = {}
        bow_boc = {}
        embedding_sent_bow_boc = {}

        date_sent = {}
        bow_date = {}
        boc_date = {}
        embedding_date = {}
        bow_sent_time = {}
        boc_sent_time = {}
        bow_boc_time = {}
        embedding_sent_time = {}
        embedding_bow_time = {}
        embedding_boc_time = {}
        bow_sent_boc_time = {}
        bow_boc_embedding_time = {}
        embedding_sent_bow_time = {}
        embedding_sent_boc_time = {}
        embedding_sent_bow_boc_time = {}


        if (os.path.exists('features/embedding_bow-'+ name +'.pkl')):
            file = open('features/embedding_bow-'+ name +'.pkl', 'rb')
            embedding_bow = pickle.load(file)

        else:
            for key in embedding_dict:
                if key in bow_dict:
                    embedding_bow[key] = np.append(embedding_dict[key], bow_dict[key])
                else:
                    print(key)

            print(len(embedding_bow))

            file = open('features/embedding_bow-'+ name +'.pkl', 'wb')
            pickle.dump(embedding_bow, file)
            file.close()

        feature_path = 'features/embedding_boc-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_boc = pickle.load(file)

        else:
            for key in embedding_dict:
                if key in boc_dict:
                    embedding_boc[key] = np.append(embedding_dict[key], boc_dict[key])
                else:
                    print(key)

            print(len(embedding_boc))

            file = open('features/embedding_boc-'+ name +'.pkl', 'wb')
            pickle.dump(embedding_boc, file)
            file.close()

        feature_path = 'features/bow_boc_embedding-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            bow_boc_embedding = pickle.load(file)

        else:
            for key in embedding_dict:
                if key in bow_dict:
                    bow_boc_embedding[key] = np.append(embedding_dict[key], bow_dict[key])
                    bow_boc_embedding[key] = np.append(bow_boc_embedding[key], boc_dict[key])
                else:
                    print(key)

            file = open('features/bow_boc_embedding-'+ name +'.pkl', 'wb')
            pickle.dump(bow_boc_embedding, file)
            file.close()

        feature_path = 'features/bow_sent_boc-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            bow_sent_boc = pickle.load(file)

        else:
            for key in bow_sent:
                bow_sent_boc[key] = np.append(bow_sent[key], boc_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(bow_sent_boc, file)
            file.close()

        feature_path = 'features/embedding_sent_bow-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_sent_bow = pickle.load(file)

        else:
            for key in embedding_sent_dict:
                embedding_sent_bow[key] = np.append(embedding_sent_dict[key], bow_dict[key])

            file = open('features/embedding_sent_bow-'+ name +'.pkl', 'wb')
            pickle.dump(embedding_sent_bow, file)
            file.close()

        feature_path = 'features/embedding_sent_boc-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_sent_boc = pickle.load(file)

        else:
            for key in embedding_sent_dict:
                embedding_sent_boc[key] = np.append(embedding_sent_dict[key], boc_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(embedding_sent_boc, file)
            file.close()

        feature_path = 'features/bow_boc-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            bow_boc = pickle.load(file)

        else:
            for key in bow_dict:
                bow_boc[key] = np.append(bow_dict[key], boc_dict[key])

            file = open('features/bow_boc-'+ name +'.pkl', 'wb')
            pickle.dump(bow_boc, file)
            file.close()

        feature_path = 'features/embedding_sent_bow_boc-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_sent_bow_boc = pickle.load(file)

        else:
            for key in embedding_sent_dict:
                embedding_sent_bow_boc[key] = np.append(embedding_sent_dict[key], bow_dict[key])
                embedding_sent_bow_boc[key] = np.append(embedding_sent_bow_boc[key], boc_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(embedding_sent_bow_boc, file)
            file.close()

        feature_path = 'features/date_sent-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            date_sent = pickle.load(file)

        else:
            for key in datetime_dict:
                date_sent[key] = np.append(datetime_dict[key], sent_dict[key])
                print(date_sent[key])

            file = open(feature_path, 'wb')
            pickle.dump(date_sent, file)
            file.close()

        feature_path = 'features/bow_date-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            bow_date = pickle.load(file)

        else:
            for key in bow_dict:
                bow_date[key] = np.append(bow_dict[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(bow_date, file)
            file.close()

        feature_path = 'features/boc_date-' + name + '.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            boc_date = pickle.load(file)

        else:
            for key in boc_dict:
                boc_date[key] = np.append(boc_dict[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(boc_date, file)
            file.close()

        feature_path = 'features/embedding_date-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_date = pickle.load(file)

        else:
            for key in embedding_dict:
                if key in bow_dict:
                        embedding_date[key] = np.append(embedding_dict[key], datetime_dict[key])
                else:
                    print(key)

            file = open(feature_path, 'wb')
            pickle.dump(embedding_date, file)
            file.close()

        feature_path = 'features/bow_sent_time-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            bow_sent_time = pickle.load(file)

        else:
            for key in bow_sent:
                bow_sent_time[key] = np.append(bow_sent[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(bow_sent_time, file)
            file.close()

        feature_path = 'features/boc_sent_time-' + name + '.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            boc_sent_time = pickle.load(file)

        else:
            for key in boc_sent:
                boc_sent_time[key] = np.append(boc_sent[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(boc_sent_time, file)
            file.close()

        feature_path = 'features/bow_boc_time-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            bow_boc_time = pickle.load(file)

        else:
            for key in bow_boc:
                bow_boc_time[key] = np.append(bow_boc[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(bow_boc_time, file)
            file.close()

        feature_path = 'features/embedding_sent_time-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_sent_time = pickle.load(file)

        else:
            for key in embedding_sent_dict:
                embedding_sent_time[key] = np.append(embedding_sent_dict[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(embedding_sent_time, file)
            file.close()

        feature_path = 'features/embedding_bow_time-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_bow_time = pickle.load(file)

        else:
            for key in embedding_bow:
                embedding_bow_time[key] = np.append(embedding_bow[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(embedding_bow_time, file)
            file.close()

        feature_path = 'features/embedding_boc_time-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_boc_time = pickle.load(file)

        else:
            for key in embedding_boc:
                embedding_boc_time[key] = np.append(embedding_boc[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(embedding_boc_time, file)
            file.close()

        feature_path = 'features/bow_sent_boc_time-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            bow_sent_boc_time = pickle.load(file)

        else:
            for key in bow_sent_boc:
                bow_sent_boc_time[key] = np.append(bow_sent_boc[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(bow_sent_boc_time, file)
            file.close()

        feature_path = 'features/bow_boc_embedding_time-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            bow_sent_boc_time = pickle.load(file)

        else:
            for key in bow_boc_embedding:
                bow_boc_embedding_time[key] = np.append(bow_boc_embedding[key], datetime_dict[key])

            file = open('features/bow_boc_embedding_time-'+ name +'.pkl', 'wb')
            pickle.dump(bow_boc_embedding_time, file)
            file.close()

        feature_path = 'features/embedding_sent_bow_time-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_sent_bow_time = pickle.load(file)

        else:
            for key in embedding_sent_bow:
                embedding_sent_bow_time[key] = np.append(embedding_sent_bow[key], datetime_dict[key])

            file = open('features/embedding_sent_bow_time-'+ name +'.pkl', 'wb')
            pickle.dump(embedding_sent_bow_time, file)
            file.close()

        feature_path = 'features/embedding_sent_boc_time-' + name + '.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_sent_boc_time = pickle.load(file)

        else:
            for key in embedding_sent_boc:
                embedding_sent_boc_time[key] = np.append(embedding_sent_boc[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(embedding_sent_boc_time, file)
            file.close()

        feature_path = 'features/embedding_sent_bow_boc_time-'+ name +'.pkl'
        if (os.path.exists(feature_path)):
            file = open(feature_path, 'rb')
            embedding_sent_bow_boc_time = pickle.load(file)

        else:
            for key in embedding_sent_bow_boc:
                embedding_sent_bow_boc_time[key] = np.append(embedding_sent_bow_boc[key], datetime_dict[key])

            file = open(feature_path, 'wb')
            pickle.dump(embedding_sent_bow_boc_time, file)
            file.close()

        # # confirm all features have same keys
        # # for key1, key2, key3 in zip(sorted(embedding_sent_dict.keys()), sorted(bow_dict.keys()), sorted(boc_dict.keys())):
        # #     embedding_feat.append(embedding_sent_dict[key1])
        # #     bow_feat.append(bow_dict[key2])
        # #     boc_feat.append(boc_dict[key3])
        # #
        # # features = [embedding_feat, bow_feat, boc_feat]
        # #
        # # # get all feature permutation
        # # all_features = self.features_pyramids(features)
        # #
        # # embedding_sent_bow = dict.fromkeys(sorted(embedding_sent_dict.keys()))
        # # embedding_sent_boc = dict.fromkeys(sorted(embedding_sent_dict.keys()))
        # # bow_boc = dict.fromkeys(sorted(embedding_sent_dict.keys()))
        # # embedding_sent_bow_boc = dict.fromkeys(sorted(embedding_sent_dict.keys()))
        # #
        # # # saving embedding+bow features
        # # for k, elem in zip(embedding_sent_bow, all_features[0]):
        # #     embedding_sent_bow[k] = elem
        # #
        # # # saving embedding+boc features
        # # for k, elem in zip(embedding_sent_boc, all_features[1]):
        # #     embedding_sent_boc[k] = elem
        # #
        # # # saving bow+boc features
        # # for k, elem in zip(bow_boc, all_features[2]):
        # #     bow_boc[k] = elem
        # #
        # # # saving embedding+bow+boc features
        # #
        # # for k, elem in zip(embedding_sent_bow_boc, all_features[3]):
        # #     embedding_sent_bow_boc[k] = elem
        #
        # return embedding_dict, bow_dict, boc_dict, sent_dict, bow_sent, boc_sent, embedding_sent_dict, \
        #        embedding_sent_bow, embedding_sent_boc, bow_boc, embedding_bow, embedding_boc, bow_sent_boc,  \
        #        bow_boc_embedding, embedding_sent_bow_boc

        return embedding_dict, bow_dict, boc_dict, sent_dict, bow_sent, boc_sent, embedding_sent_dict, \
    embedding_sent_bow, embedding_sent_boc, bow_boc, embedding_bow, embedding_boc, bow_sent_boc, \
    bow_boc_embedding, embedding_sent_bow_boc, datetime_dict, date_sent, bow_date, boc_date, embedding_date, \
    bow_sent_time, boc_sent_time, bow_boc_time, \
    embedding_sent_time, embedding_bow_time, embedding_boc_time, bow_sent_boc_time, bow_boc_embedding_time, \
    embedding_sent_bow_time, embedding_sent_boc_time, embedding_sent_bow_boc_time

        # return bow_boc_embedding, embedding_sent_bow_time, bow_boc_embedding_time
        #return embedding_bow, embedding_boc


#  --- Test Feature Pyramids to generate all possible features ----
def main():
    fe = Features()

    # <key, value> representation of features.
    # embedding_bow, embedding_boc, bow_boc, embedding_bow_boc = fe.get_all_features()


if __name__ == '__main__':
    main()