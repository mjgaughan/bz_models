import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import treebank
from difflib import SequenceMatcher
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger')
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
# param len/word len
def param_traits(param):
    param_split = param.split()
    param_name = param_split[-1]
    return (len(param_name))
#TODO: high value token and similarity matching
def action_on_sub(definition, param):
    #getting the function name
    func_name = definition[:definition.find("(")]
    func_name = func_name.split()[-1]
    #comparing the param and the func name, this is somewhat of an arbitrary pattern matcher, not attached to it
    #if there is a better option, can find, just felt like would show it
    # https://docs.python.org/3/library/difflib.html
    matcher = SequenceMatcher(None,func_name, param)
    similarity = matcher.ratio()
    if similarity > 0.25:
        func_name_bagged = lemmatize_bagged(func_name)
        func_name_pos_tagged = nltk.pos_tag(func_name_bagged)
        print(func_name_pos_tagged)
#TODO: group function tokens?

def main_analysis(definition, param):
    get_param_location(definition, param)
    #bagged = bow_def(definition)
    #lemmatized_bag = lemmatize(bagged)
    #return lemmatized_bag
    
    #get_return_value(definition)

