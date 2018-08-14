# TREC-IS

##### Install the following dependencies before using the ```Feature_Extractor``` class: <br>
- [spacy](https://spacy.io/usage/models#section-install) :
```
pip install -U spacy 
python -m spacy download en
python -m spacy download en_core_web_lg
```
- [nltk](https://www.nltk.org/install.html) <br>
```pip install -U nltk ``` <br>
Enter python shell and then download all the nltk packages. 
```
>> import nltk
>> nltk.download( )

```
- [scikit-learn](http://scikit-learn.org/stable/install.html)
```
pip install -U scikit-learn
```
- [emoji](https://pypi.org/project/emoji/)
```
pip install emoji --upgrade
```

- [textblob](https://textblob.readthedocs.io/en/dev/)
```
pip install -U textblob
python -m textblob.download_corpora

```
- [Word Embedding](https://www.fredericgodin.com/software/)

We used a pre-trained word embedding model trained on a tweets dataset. It couldn't be pushed in the 
repository. 
- Download it from this link https: https://www.fredericgodin.com/software/ 
- Save into 'data' folder. 

##### For using Bag-of-Concepts, first create an account on [BabelNet](https://babelnet.org/register) and after logging in, fill the form as mentioned [here](http://babelfy.org/guide#HowcanIincreasetheBabelcoinsdailylimit?) to increase the daily limit. 
Add the unique API key as 'babelnet_key' in secrets.py and then you're ready to go.!    
