import sys

import nltk
from nltk.stem.porter import *
#from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import xml.etree.cElementTree as ET
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from zipfile import ZipFile
import os

PARTIALS = False


def gettext(xmltext):
    """
    Parse xmltext and return the text from <title> and <text> tags
    """
    xmltext = xmltext.encode('ascii', 'ignore') # ensure there are no weird char
    
    # PUT XML INTO ELEMENT TREE
    xmltext = ET.fromstring(xmltext)
    tree = ET.ElementTree(xmltext)
        
    textstr = str()
    
    # APPEND TEXT UNDER TITLE TAGS
    for elem in tree.iterfind('title'):
        textstr = textstr + elem.text 
        
    # APPEND TEXT UNDER TEXT TAGS        
    for elem in tree.iterfind('.//text/*'):
        textstr = textstr + ' ' + elem.text        
    
    # RETURN STRING OF TEXT FROM TITLE AND TEXT TAGS OF SPECIFIED DOC
    return textstr


def tokenize(text): 
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3, strip digits.
    """
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 2]  # ignore a, an, to, at, be, ...
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    
    return tokens
    

def stemwords(words): 
    """
    Given a list of tokens/words, return a new list with each word
    stemmed using a PorterStemmer.
    """
    stemmer = PorterStemmer()
    stemlist = [stemmer.stem(x) for x in words]
    
    return stemlist


def tokenizer(text): 
    return stemwords(tokenize(text))


def compute_tfidf(corpus): 
    """
    Create and return a TfidfVectorizer object after training it on
    the list of articles pulled from the corpus dictionary. Meaning,
    call fit() on the list of document strings, which figures out
    all the inverse document frequencies (IDF) for use later by
    the transform() function. The corpus argument is a dictionary 
    mapping file name to xml text.
    """
    # CREATE TRANSFORM OBJECT
    tfidf = TfidfVectorizer(input='content',
                        analyzer='word',
                        preprocessor=gettext,
                        tokenizer=tokenizer,
                        stop_words='english', # even more stop words
                        decode_error = 'ignore')
    
    text = list(corpus.values())
    
    for loc,x in enumerate(text):
        if x == '':
            text.pop(loc)

        else:
            pass
    
    # INPUT FILE STRINGS AND COMPUTE IDF
    tfidf = tfidf.fit(text)
    
    return tfidf
    

def summarize(tfidf, text, n): 
                                
    """
    Given a trained TfidfVectorizer object and some XML text, return
    up to n (word,score) pairs in a list. Discard any terms with
    scores < 0.09.
    """
    summary = tfidf.transform([text])
    
    # GET INDEX OF NON ZERO ELEMENTS IN MATRIX
    index = summary.nonzero()

    # GET LIST OF WORDS IN VECTORIZER OBJECT
    terms = tfidf.get_feature_names()
    
    scorelist = []
    
    # ITERATE THROUGH WORD INDEXES OF NON-ZERO ELEMENTS AND GET CORRESPONDING WORD AND TFIDF SCORE
    for x in index[1]:
        word = terms[x]
        score = summary[0,x]
        
        tup = (word, score)
        
        scorelist.append(tup)
        
    scorelist.sort(key=lambda x:x[1], reverse = True)
    
    finallist = []
    
    # ITERATE THROUGH FIRST n ELEMENTS OF LIST AND IF SCORE GREATER THAN .09, PRINT    
    for x in scorelist[:n]:
        if x[1] >= 0.09:
            finallist.append(x)
        else:
            pass
    
    return finallist


def load_corpus(zipfilename): 
    """
    Given a zip file containing root directory reuters-vol1-disk1-subset
    and a bunch of *.xml files, read them from the zip file into
    a dictionary of (filename,xmltext) associations. Use namelist() from
    ZipFile object to get list of xml files in that zip file.
    Convert filename reuters-vol1-disk1-subset/foo.xml to foo.xml
    as the keys in the dictionary. The values in the dictionary are the
    raw XML text from the various files.
    """
    corpusdict = {}
    
    # GET A LIST OF PATHS TO FILES IN ZIP DIR
    with ZipFile(zipfilename, 'r') as obj:
        pathlist = obj.namelist()
            
    zip = ZipFile(zipfilename)
    
    # FOR EACH FILE PATH, GET RAW TEXT OF FILE AND NAME OF FILE, AND ADD BOTH TO DICT
    for path in pathlist:
        rawtext = zip.read(path) # BYTES
        rawtext = rawtext.decode('utf-8') # CONVERT BYTES TO STRING
        
        filename = os.path.basename(path)
        
        corpusdict[filename] = rawtext
    
    # RETURN DICT WITH 'KEYS:FILE NAMES' AND 'VALUES: RAW TEXT OF FILES'
    return corpusdict
        