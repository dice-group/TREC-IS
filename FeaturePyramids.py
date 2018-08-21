import itertools as it
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

    # sample lists
    A = [[1, 2, 3], ['A', 'B'], [1.5, 2.5, 3.5]]
    B = [[4, 5, 6, 7], ['C', 'D'], [4.5, 5.5, 6.5]]
    C = [[8, 8, 8], ['E', 'F'], [7.5, 8.5, 9.5]]

    features = [A, B, C]

    all_features = fe.get_all_Features(features)

    for k in all_features:
        print(all_features[k])

if __name__ == '__main__':
    main()