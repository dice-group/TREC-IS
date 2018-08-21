import itertools as it
import pickle


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
        embedding_dict = pickle.load(open('features/embedding_sentiment.pkl', 'rb'))
        bow_dict = pickle.load(open('features/bow.pkl', 'rb'))
        boc_dict = pickle.load(open('features/boc_OHE.pkl', 'rb'))

        embedding_feat = []
        bow_feat = []
        boc_feat = []

        # confirm all features have same keys
        for key1, key2, key3 in zip(sorted(embedding_dict.keys()), sorted(bow_dict.keys()), sorted(boc_dict.keys())):
            embedding_feat.append(embedding_dict[key1])
            bow_feat.append(bow_dict[key2])
            boc_feat.append(boc_dict[key3])

        features = [embedding_feat, bow_feat, boc_feat]

        # get all feature permutation
        all_features = self.features_pyramids(features)

        embedding_bow = dict.fromkeys(sorted(embedding_dict.keys()))
        embedding_boc = dict.fromkeys(sorted(embedding_dict.keys()))
        bow_boc = dict.fromkeys(sorted(embedding_dict.keys()))
        embedding_bow_boc = dict.fromkeys(sorted(embedding_dict.keys()))

        # saving embedding+bow features
        for k, elem in zip(embedding_bow, all_features[0]):
            embedding_bow[k] = elem

        # saving embedding+boc features
        for k, elem in zip(embedding_boc, all_features[1]):
            embedding_boc[k] = elem

        # saving bow+boc features
        for k, elem in zip(bow_boc, all_features[2]):
            bow_boc[k] = elem

        # saving embedding+bow+boc features

        for k, elem in zip(embedding_bow_boc, all_features[3]):
            embedding_bow_boc[k] = elem

        return embedding_bow, embedding_boc, bow_boc, embedding_bow_boc


#  --- Test Feature Pyramids to generate all possible features ----
def main():
    fe = Features()

    # <key, value> representation of features.
    embedding_bow, embedding_boc, bow_boc, embedding_bow_boc = fe.get_all_features()


if __name__ == '__main__':
    main()
