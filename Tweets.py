import pandas as pd

class Tweets:

    def __init__(self, trainingData=None, fullData=None):
        '''
        :param TREC-IS trainingData: (postID,categories,indicatorTerms,priority)
        :param fullData: as standard tweets attributes
        https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.html
        '''

        self.TREC_data = trainingData
        self.full_data = fullData

    def clean(self):

        return 'Todo'

    def load_trainingData(self):

        trainingData=pd.DataFrame()



