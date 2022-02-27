import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import treebank

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
breaks = ["_", "(", ")"]
stripped_punctuation = [",", '.', ";", ":", '“', '$', '`']


def bow_def(definition):
    working_cor = ""
    for letter in definition:
        if letter in breaks:
            working_cor += " "
        else:
            working_cor += letter
    bag = nltk.word_tokenize(working_cor)
    return bag 

def lemmatize(bag):
    lemmatizer = WordNetLemmatizer()
    minimized_bag = []
    for word in bag:
        if word not in stripped_punctuation:
            minimized_bag.append(lemmatizer.lemmatize(word))
    print(minimized_bag)
    return minimized_bag

#TODO: grabbing text from the entire function def

def main_analysis(definition):
    bagged = bow_def(definition)
    lemmatized_bag = lemmatize(bagged)
    return lemmatized_bag


    

