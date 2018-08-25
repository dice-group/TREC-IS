import itertools as it
import pickle
import numpy as np

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

    def get_all_features(self):

        # loading saved features
        sent_dict = pickle.load(open('features/sentiment.pkl', 'rb'))
        bow_sent =  pickle.load(open('features/bow_sentiment.pkl', 'rb'))
        boc_sent = pickle.load(open('features/boc_sentiment.pkl', 'rb'))
        embedding_dict = pickle.load(open('features/embedding_features.pkl', 'rb'))
        embedding_sent_dict = pickle.load(open('features/embedding_sentiment.pkl', 'rb'))
        bow_dict = pickle.load(open('features/bow.pkl', 'rb'))
        boc_dict = pickle.load(open('features/boc_OHE.pkl', 'rb'))

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

        for key in embedding_dict:
            if key in bow_dict:
                embedding_bow[key] = np.append(embedding_dict[key], bow_dict[key])
            else:
                print(key)

        for key in embedding_dict:
            if key in boc_dict:
                embedding_boc[key] = np.append(embedding_dict[key], boc_dict[key])
            else:
                print(key)

        for key in embedding_dict:
            if key in bow_dict:
                bow_boc_embedding[key] = np.append(embedding_dict[key], bow_dict[key])
                bow_boc_embedding[key] = np.append(bow_boc_embedding[key], boc_dict[key])
            else:
                print(key)

        for key in bow_sent:
            bow_sent_boc[key] = np.append(bow_sent[key], boc_dict[key])

        for key in embedding_sent_dict:
            embedding_sent_bow[key] = np.append(embedding_sent_dict[key], bow_dict[key])

        for key in embedding_sent_dict:
            embedding_sent_bow[key] = np.append(embedding_sent_dict[key], bow_dict[key])

        for key in embedding_sent_dict:
            embedding_sent_boc[key] = np.append(embedding_sent_dict[key], boc_dict[key])

        for key in bow_dict:
            bow_boc[key] = np.append(bow_dict[key], boc_dict[key])

        for key in embedding_sent_dict:
            embedding_sent_bow_boc[key] = np.append(embedding_sent_dict[key], bow_dict[key])
            embedding_sent_bow_boc[key] = np.append(embedding_sent_bow_boc[key], boc_dict[key])

        # confirm all features have same keys
        # for key1, key2, key3 in zip(sorted(embedding_sent_dict.keys()), sorted(bow_dict.keys()), sorted(boc_dict.keys())):
        #     embedding_feat.append(embedding_sent_dict[key1])
        #     bow_feat.append(bow_dict[key2])
        #     boc_feat.append(boc_dict[key3])
        #
        # features = [embedding_feat, bow_feat, boc_feat]
        #
        # # get all feature permutation
        # all_features = self.features_pyramids(features)
        #
        # embedding_sent_bow = dict.fromkeys(sorted(embedding_sent_dict.keys()))
        # embedding_sent_boc = dict.fromkeys(sorted(embedding_sent_dict.keys()))
        # bow_boc = dict.fromkeys(sorted(embedding_sent_dict.keys()))
        # embedding_sent_bow_boc = dict.fromkeys(sorted(embedding_sent_dict.keys()))
        #
        # # saving embedding+bow features
        # for k, elem in zip(embedding_sent_bow, all_features[0]):
        #     embedding_sent_bow[k] = elem
        #
        # # saving embedding+boc features
        # for k, elem in zip(embedding_sent_boc, all_features[1]):
        #     embedding_sent_boc[k] = elem
        #
        # # saving bow+boc features
        # for k, elem in zip(bow_boc, all_features[2]):
        #     bow_boc[k] = elem
        #
        # # saving embedding+bow+boc features
        #
        # for k, elem in zip(embedding_sent_bow_boc, all_features[3]):
        #     embedding_sent_bow_boc[k] = elem

        return embedding_dict, bow_dict, boc_dict, sent_dict, bow_sent, boc_sent, embedding_sent_dict, \
               embedding_sent_bow, embedding_sent_boc, bow_boc, embedding_bow, embedding_boc, bow_sent_boc,  \
               bow_boc_embedding, embedding_sent_bow_boc



#  --- Test Feature Pyramids to generate all possible features ----
def main():
    fe = Features()

    # <key, value> representation of features.
    embedding_bow, embedding_boc, bow_boc, embedding_bow_boc = fe.get_all_features()


if __name__ == '__main__':
    main()
