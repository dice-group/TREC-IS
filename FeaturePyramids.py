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

    def get_all_Features(self, features_list):
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


#  --- Test Feature Pyramids to generate all possible features ----
def main():
    fe = Features()

    embedding_dict = pickle.load(open('features/embedding_sentiment.pkl', 'rb'))
    bow_dict = pickle.load(open('features/bow.pkl', 'rb'))
    boc_dict = pickle.load(open('features/boc_OHE.pkl', 'rb'))

    embedding_feat = []
    bow_feat = []
    boc_feat = []

    for key1, key2, key3 in zip(sorted(embedding_dict.keys()), sorted(bow_dict.keys()), sorted(boc_dict.keys())):
        embedding_feat.append(embedding_dict[key1])
        bow_feat.append(bow_dict[key2])
        boc_feat.append(boc_dict[key3])

    features = [embedding_feat, bow_feat, boc_feat]

    all_features = fe.get_all_Features(features)

    # iterate features of embedding+bow
    for elem in all_features[0]:
        print(elem)

if __name__ == '__main__':
    main()