from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class Evaluation:
    '''
    Evaluating the classification performance as F1 score (Micro averaged precision, recall and accuracy)
    '''

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        self.f1_score = f1_score(self.y_true, self.y_pred, average='micro')

        self.accuracy_score = accuracy_score(self.y_true, self.y_pred)
