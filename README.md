# TREC-IS

##### Install the following dependencies before using the ```Feature_Extractor``` class: <br>
- [spacy](https://spacy.io/usage/models#section-install) :
```
pip install -U spacy 
python -m spacy download en
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
```