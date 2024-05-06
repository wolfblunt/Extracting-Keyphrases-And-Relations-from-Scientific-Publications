
from sklearn.feature_extraction.text import CountVectorizer

def generateCandidates(doc, minLength, maxLength):
    n_gram_range = (minLength, maxLength)
    stop_words = "english"
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names_out()
    return candidates

