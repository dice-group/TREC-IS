from contractions import CONTRACTION_MAP
import string, spacy, nltk, re
import emoji
from secrets import babelnet_key
from difflib import get_close_matches
import requests

babelfy_url = 'https://babelfy.io/v1/disambiguate'
babelnet_url = 'https://babelnet.io/v5/getSynset'
key = babelnet_key
lang = 'EN'

nlp = spacy.load('en')

class Helper_FeatureExtraction:

    def extract_synsets_from_babelfy(self, text):
        '''
        for extracting synsetIDs from text recognition and disambiguation
        :param text: string
        :return: ({concept_mentions : synsetIDs}, {concept_mentions : (score, coherence_score, global_score)})
        '''
        params = {
            'text': text,
            'lang': lang,
            'key': key
        }
        response = requests.get(babelfy_url, params=params, headers={'Accept-encoding': 'gzip'})
        data = response.json()

        synsetIDs = []
        mentions = []
        scores = []
        for result in data:
            # retrieving char fragment
            charFragment = result.get('charFragment')
            cfStart = charFragment.get('start')
            cfEnd = charFragment.get('end')
            mentions.append(text[cfStart:cfEnd + 1])

            # retrieving BabelSynset ID
            synsetId = result.get('babelSynsetID')
            synsetIDs.append(synsetId)

            # retrieving the scores
            coherence_score = result.get('coherenceScore')
            global_score = result.get('globalScore')
            score = result.get('score')
            scores.append((score, coherence_score, global_score))

        synsetDict = dict(zip(mentions, synsetIDs))
        scoreDict = dict(zip(mentions, scores))
        return (synsetDict, scoreDict)

    def extract_concepts_from_babelnet(self, text):
        '''
        Uses the synsets returned by Babelfy to extract categories for each disambiguated term.
        Only the categories (for each term) that are most similar to the 'mainSense' are considered.
        :param text: string
        :return: a list of concepts from the text
        '''
        synsetObj = self.extract_synsets_from_babelfy(text)
        synsetDict = synsetObj[0]
        concepts = []
        for k in synsetDict.keys():
            params = {
                'id': synsetDict[k],
                'key': key
            }
            r = requests.get(babelnet_url, params=params, headers={'Accept-encoding': 'gzip'})
            data = r.json()
            main_sense = data['mainSense']
            categories = data['categories']

            category_terms = []
            for category in categories:
                category_terms.append(category.get('category'))

            maxscore = 0
            main_sense = main_sense.replace('#', '').replace('n1', '') #mainSense specific preprocessing

            # if difflib returns an empty list, use cosine similarity from spacy
            finalCategory = get_close_matches(main_sense, category_terms)
            for cat in finalCategory:
                concepts.append(cat)

            if not finalCategory:
                for term in category_terms:
                    w1 = nlp(term)
                    w2 = nlp(main_sense)
                    score = w1.similarity(w2)
                    maxscore = max(score, maxscore)
                    if maxscore == score:
                        concepts.append(term)

        return concepts

    def hashtag_pipe(self, text):
        '''
        Processes hashtags as one word.
        Add this custom pipeline to spacy's nlp pipeline before running it on the desired text.
        :param text:
        :return:
        '''
        merged_hashtag = False
        while True:
            for token_index,token in enumerate(text):
                if token.text == '#':
                    if token.head is not None:
                        start_index = token.idx
                        end_index = start_index + len(token.head.text) + 1
                        if text.merge(start_index, end_index) is not None:
                            merged_hashtag = True
                            break
            if not merged_hashtag:
                break
            merged_hashtag = False
        return text

    def remove_stopwords_and_punctuations(self, text, nlp):
        '''
        text = "It's going be a rainy week for Davao. #PabloPH http://t.co/XnObb62J"
        output = "It going rainy week Davao PabloPH http://t.co/XnObb62J"
        :param text:
        :return:
        '''
        stopwords= nltk.corpus.stopwords.words('english')
        stopwords.extend(string.punctuation)
        stopwords.append('')
        customize_spacy_stop_words = ["rt", "'ve", "n't", "\n", "'s"]
        for w in customize_spacy_stop_words:
            nlp.vocab[w].is_stop = True
        parsed_text = nlp(text)
        tokens = [(token.text) for token in parsed_text if not str(token) in stopwords and not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    def lemmatize_text(self, text, nlp):
        '''
        text = "It's going be a rainy week for Davao. #PabloPH http://t.co/XnObb62J"
        lem_text = "It be go be a rainy week for davao . # pabloph http://t.co/xnobb62j"
        :return:
        '''
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text

    def expand_contractions(self, text, contraction_mapping=CONTRACTION_MAP):
        '''
        Eg: text = "It's going be a rainy week for Davao. #PabloPH http://t.co/XnObb62J"
        exapnded_text = "It is going be a rainy week for Davao. #PabloPH http://t.co/XnObb62J"
        :param contraction_mapping:
        :return: text with expanded words
        '''
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

    def remove_username(self, text):
        text = re.sub('@[^\s]+', '.', text)
        return text

    def remove_url(self, text):
        text = re.sub(r"http\S+", "", text)
        return text

    def remove_special_symbols(self, text):
        '''
        removes arabic, tamil, latin symbols and dingbats
        :param text:
        :return:
        '''
        special_symbols = re.compile(r"[\u0600-\u06FF\u0B80-\u0BFF\u25A0-\u25FF\u2700-\u27BF]+", re.UNICODE)
        text = special_symbols.sub('', text)
        return text

    def emoji_to_text(self, text):
        text = emoji.demojize(text)
        text = text.replace("::", " ") #for emojis that don't have space between them
        return text

    def remove_emojis(self, text):
        emoji_list = [char for char in text if char in emoji.UNICODE_EMOJI]
        clean_text = ' '.join([tok for tok in text.split() if not any(char in tok for char in emoji_list)])
        return clean_text

    def remove_numbers(self, text):
        text = re.sub(r'\d+', '', text)
        return text

    def normalize_tweet(self, text, nlp, demojize_text= True, special_symbol_removal= True, emoji_removal= False, contraction_expansion=True, lemmatization= True, remove_stopwords = True, hashtags_intact= True, url_removal= True, number_removal=True, username_removal= True ):

        if emoji_removal:
            text = self.remove_emojis(text)
        elif demojize_text:
            text = self.emoji_to_text(text)

        if special_symbol_removal:
            text = self.remove_special_symbols(text)

        if contraction_expansion:
            text = self.expand_contractions(text)

        text = text.strip() # remove whitespaces
        text = re.sub(' +', ' ', text)  # remove extra whitespace

        if username_removal:
            text = self.remove_username(text)

        if url_removal:
            text = self.remove_url(text)

        if number_removal:
            text = self.remove_numbers(text)

        if hashtags_intact and 'hashtag_pipe' not in nlp.pipe_names:
            nlp.add_pipe(self.hashtag_pipe)

        if lemmatization:
            text = self.lemmatize_text(text, nlp)

        if remove_stopwords:
            text = self.remove_stopwords_and_punctuations(text, nlp)

        return text

    def extract_keywords_from_tweets(self, input_dataframe):
        norm_tweets = []
        for _, row in input_dataframe.iterrows():
            norm_text = self.normalize_tweet(str(row['text']).lower(), nlp) #lowercased input
            norm_tweets.append(norm_text)

        return norm_tweets

    def include_indicatorTerms_in_tweets(self, input_dataframe):
        norm_tweets = []
        for _, col in input_dataframe.iterrows():
            norm_text = self.normalize_tweet(str(col['text']).lower(), nlp, lemmatization=True) #lowercased input, lemmatization gives better results with MultinomialNB only
            if col['indicatorTerms']:
                norm_text += ' '.join(col['indicatorTerms'])
            norm_tweets.append(norm_text)

        return norm_tweets