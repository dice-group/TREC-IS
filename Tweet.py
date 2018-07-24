class Tweet:
    def __init__(self, id=None, text=None, metadata=None, priority=None, indicatorTerms=None, categories=None):
        self.id = id
        self.text = text
        self.metadata = metadata
        self.priority = priority
        self.indicatorTerms = indicatorTerms
        self.categories = categories


    def add_tweets_data(self, text=None, metadata=None):
        '''
        This function to combine tweet info (full_text,
        metadata) with tweet's trec-is data (tweet_priority,
        tweet_indicatorTerms, tweet_categories)
        :param text:
        :param metadata:
        :return:
        '''
        self.text = text
        self.metadata = metadata
