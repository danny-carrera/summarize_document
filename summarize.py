from tfidf import *
import sys

zipfilename = sys.argv[1]
summarizefile = sys.argv[2]

# get dict with key=file name, value=file's raw text
corpus = load_corpus(zipfilename)
corpus.pop('')

# train vectorizer object on all files
tfidf = compute_tfidf(corpus)

# get raw text of specified file
text = corpus[summarizefile]

# get the top n words with the highest tfidf score
n = 20
summary = summarize(tfidf, text, n)
   
for word in summary:
    print(word[0], f'{round(word[1], 3):.3f}')
