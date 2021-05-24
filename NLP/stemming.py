import nltk as nltk
import nltk.stem as stem
import wordcloud
import matplotlib.pyplot as plt
import pandas as pd

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
stemmer = stem.PorterStemmer()

plurals = ["caresses", "dies", "flies", "humbled", "historical", "goes", "feet", "wolves"]

singles = [stemmer.stem(word) for word in plurals]
print(singles)

lemma = stem.WordNetLemmatizer()
print([lemma.lemmatize(word) for word in plurals])


sentences = nltk.sent_tokenize("aa. bbb abc aaa")
print(sentences)
