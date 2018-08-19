import itertools as it


class Features:

    def generate_Features(self, features):
        '''
        :param features: list of features extracted from tweets
        :return: a list of all possible combination of features
        '''

        all_the_features = []
        for r in range(1, len(features) + 1):
            all_the_features = all_the_features + list(it.combinations(features, r))

        featurePyramids = []

        for feature in all_the_features:
            feature_set = []
            for t in feature:
                feature_set += list(t)
            featurePyramids.append(feature_set)

        return featurePyramids


#  --- Test Feature Pyramids to generate all possible features ----
def main():
    fe = Features()

    # sample lists
    A = [1, 2, 3]
    B = [4, 5, 6, 7]
    C = [8, 8, 8]
    features = [A, B, C]

    featurePyramids = fe.generate_Features(features)
    print(featurePyramids)


if __name__ == '__main__':
    main()
