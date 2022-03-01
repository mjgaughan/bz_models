import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import treebank

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
breaks = ["_", "(", ")"]
stripped_punctuation = [",", '.', ";", ":", '“', '$', '`']


def lemmatize_bagged(definition):
    working_cor = ""
    for letter in definition:
        if letter in breaks:
            working_cor += " "
        else:
            working_cor += letter
    bag = nltk.word_tokenize(working_cor)
    lemmatizer = WordNetLemmatizer()
    minimized_bag = []
    for word in bag:
        if word not in stripped_punctuation:
            minimized_bag.append(lemmatizer.lemmatize(word))
    return minimized_bag

# get function return value
def get_return_value(definition):
    definition_split = definition.split()
    return definition_split[0]

# location in the param list
def get_param_location(definition, param):
    begin = definition.find("(")
    end = definition.find(")")
    parameters = definition[begin:end]
    parameter_list = parameters.split(",")
    for index in range(len(parameter_list)):
        if param in parameter_list[index]:
            #print(index)
            return index
    #print(parameter_list)
    return -1
#TODO: param len/word len
def param_traits(param):
    param_split = param.split()
    param_name = param_split[-1]
    return (len(param_name))
#TODO: high value token and similarity matching
def action_on_sub(definition, param):
    print("hi")
#TODO: group function tokens?

def main_analysis(definition, param):
    get_param_location(definition, param)
    #bagged = bow_def(definition)
    #lemmatized_bag = lemmatize(bagged)
    #return lemmatized_bag
    
    #get_return_value(definition)

