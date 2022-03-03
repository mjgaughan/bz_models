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

'''
this function takes the function definition and spits out a lemmatized bag of words
'''
def lemmatize_bagged(definition):
    working_cor = ""
    for letter in definition:
        #there are some special break characters used to break up words and it's easier for tokenizing if they're just spaces
        if letter in breaks:
            working_cor += " "
        else:
            working_cor += letter
    #tokenize the words into a bag
    bag = nltk.word_tokenize(working_cor)
    #lemmatize them
    lemmatizer = WordNetLemmatizer()
    minimized_bag = []
    #also toss out any punctuation
    for word in bag:
        if word not in stripped_punctuation:
            minimized_bag.append(lemmatizer.lemmatize(word))
    return minimized_bag

# get function return value
def get_return_value(definition):
    definition_split = definition.split()
    return definition_split[0]

# location in the param list
# the location of the parameter in the list of parameter in the function definition may be important
def get_param_location(definition, param):
    begin = definition.find("(")
    end = definition.find(")")
    parameters = definition[begin:end]
    parameter_list = parameters.split(",")
    for index in range(len(parameter_list)):
        if param in parameter_list[index]:
            return index
    return -1
# the length of the syntax of the parameter name
def param_traits(param):
    param_split = param.split()
    param_name = param_split[-1]
    return (len(param_name))
'''
high value token and similarity matching
the thought is that if the function name refers to the parameter name then the parameter is key to the function
furthermore, if the function name contains an action word that infers mutability, that could be a key clue
'''
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
        #there are a couple different ways to do this, OTS PoS taggers didn't work super well so just having it check against a preselected wordlist atm
        func_name_bagged = lemmatize_bagged(func_name)
        func_name_pos_tagged = nltk.pos_tag(func_name_bagged)
        print(func_name_pos_tagged)


def main_analysis(definition, param):
    get_param_location(definition, param)


