from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

stop = stopwords.words('english')
wnl = WordNetLemmatizer()

def low_case(word_array):
    word_array = word_array.str.lower()
    return word_array
def stopwords(word_array):
    word_array = word_array.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))
    return word_array
def punctuation(word_array):
    word_array = word_array.str.replace('[^\w\s]' ,'')
    word_array = word_array.str.replace('\n','')
    word_array = word_array.str.replace('  ' ,' ')
    return word_array
def numbers(word_array):
    word_array = word_array.apply(lambda x : re.sub(r'\d+', '', x))
    return word_array
def tokenize(word_array):
    word_array = word_array.apply(word_tokenize)
    return word_array
def lemm_words(word_array):
    word_array = word_array.apply(lambda x: " ".join([wnl.lemmatize(word) for word in x.split()]))
    return word_array

def clean_pipeline(word_array):
    word_array = low_case(word_array)
    word_array = stopwords(word_array)
    word_array = punctuation(word_array)
    word_array = lemm_words(word_array)
    word_array = numbers(word_array)
    word_array = tokenize(word_array)
    return word_array
