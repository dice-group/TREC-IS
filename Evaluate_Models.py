
from markdown2 import Markdown
from sklearn.preprocessing import scale, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import matplotlib as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score, make_scorer
# Variables for average classification report
originalclass = []
predictedclass = []

class ModelEvaluation:


    def __init__(self,  X, y, feature_name):
        '''
        :param X: training features
        :param y: categories
        '''

        # Scale and normalize training features
        self.X = scale(X)
        self.X = normalize(X, norm='l2')
        # self.X = X
        self.y = y
        self.feature_name = feature_name

    def classification_report_with_accuracy_score(self, y_true, y_pred):

        originalclass.extend(y_true)
        predictedclass.extend(y_pred)
        return accuracy_score(y_true, y_pred)  # return accuracy score


    def run_evaluation(self):

        with open('evaluation/performance_report.md', 'a') as f:

            f.write('------' + self.feature_name + '--------')
            f.write('\n')

            names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Linear SVM (squared loss)",  "Logistic Regression",
                     "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                     "Naive Bayes", "QDA"]

            models = [
                KNeighborsClassifier(3),
                SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1),
                LinearSVC(random_state=42),
                LogisticRegression(random_state=42, solver='newton-cg'),
                GaussianProcessClassifier(1.0 * RBF(1.0)),
                DecisionTreeClassifier(max_depth=5, random_state=42),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                MLPClassifier(alpha=1),
                AdaBoostClassifier(),
                GaussianNB(),
                # QuadraticDiscriminantAnalysis()
                 ]

            kf = 2

            entries = []
            for name, model in zip(names, models):
                scores = cross_val_score(model, self.X, self.y, scoring=make_scorer(self.classification_report_with_accuracy_score), cv=kf)
                # Average values in classification report for all folds in a K-fold Cross-validation
                print(name)
                # f.write(name)
                # f.write(classification_report(originalclass, predictedclass))

                for fold_idx, accuracy in enumerate(scores):
                    entries.append((name, fold_idx, accuracy))

            cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
            mean_acc = cv_df.groupby('model_name').accuracy.mean()
            f.write(str(mean_acc))
            f.write('\n\n')
            print(mean_acc)

            #-----------visualization of models and accuracies---------
            # sns.barplot(x='model_name', y = 'accuracy', data=cv_df)
            # plt.show()







