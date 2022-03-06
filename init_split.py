import csv
import nltk
import math
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typing as T
from sklearn.utils import resample
from collections import defaultdict
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer

from text_analysis import lemmatize_bagged, main_analysis, get_return_value, get_param_location, param_traits, action_on_sub
from body_analysis import body_len, find_left_invoke

ys = []
prototypes = []
RANDOM_SEED = 1999
numberer = DictVectorizer(sparse=False)
scaling = StandardScaler()
#hypothesis: while the model will be more efficient than other approaches, it will have difficulty labeling in such a constrained context

def data_pp(data, body):
    with open(data) as fp:
        location = 0
        rows = csv.reader(fp)
        header = next(rows)
        return_values = []
        for row in rows:
            datapoint = {}
            for (column_name, column_value) in zip(header,row):
                datapoint[column_name] = column_value
            print(datapoint)
            #formatting data point binary in 0/1
            if datapoint['immutable'] == "False":
                immutable = 0
            else:
                immutable = 1
            del datapoint['immutable']
            #ys.append(immutable)
            target_param = ""
            #finding the targeted param
            for i in range(10):
                current_param_entry = datapoint["a" + str(i)]
                if current_param_entry != "ignore" and current_param_entry != "u":
                    target_param = current_param_entry
                    break
            #creating new features
            new_datapoint= {"func_prototype": datapoint["func_prototype"], "target_param": target_param, "lemmatized_bow": lemmatize_bagged(datapoint["func_prototype"]) }
            #new_datapoint = {"target_param" : target_param, "func_prototype": datapoint["func_prototype"]}
            new_datapoint["func_return_value"] =  get_return_value(datapoint["func_prototype"])
            return_values.append(new_datapoint["func_return_value"])
            new_datapoint["parameter_location"] = get_param_location(datapoint["func_prototype"], target_param)
            new_datapoint["param_name_len"] = param_traits(target_param)
            new_datapoint["immutable"] = immutable
            if body:
                new_datapoint["body_length"] = body_len(datapoint["func_body_text"])
                new_datapoint["left_of_eq"] = find_left_invoke(datapoint["func_body_text"], target_param)

            #new_datapoint["relevant_action"] = action_on_sub(datapoint["func_prototype"], target_param)
            #adding to big set
            prototypes.append(new_datapoint)
            #for testing
            action_on_sub(datapoint["func_prototype"], target_param) 
            location += 1
            if location == 100:
                break
        #print(prototypes)
        le = LabelEncoder()
        le.fit(return_values)
        encoded_return_values = le.transform(return_values)
        for i in range(len(prototypes)):
            prototypes[i]["func_return_value"] = encoded_return_values[i]
        return prototypes


def split_data(xs):
    #if generating more features, may need multiple splitting things 
    f_tv, f_test = train_test_split(
        xs,
        train_size=0.75,
        shuffle=True,
        random_state=RANDOM_SEED
    )
    f_train, f_validate = train_test_split(
        f_tv, train_size=0.66, shuffle=False, random_state=RANDOM_SEED
    )
    
    func_to_column = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, stop_words="english", max_df=0.5
    )

    func_to_column.fit(f_train["func_prototype"].to_list())
    
    f_train_bow = func_to_column.transform(f_train["func_prototype"].to_list())
    f_validate_bow = func_to_column.transform(f_validate["func_prototype"].to_list())
    f_test_bow = func_to_column.transform(f_test["func_prototype"].to_list())

    del f_train["lemmatized_bow"]
    del f_validate["lemmatized_bow"]
    del f_test["lemmatized_bow"]
    #f_train["func_col"] = f_train_bow

    #pd.options.display.max_colwidth = 200
    print(f_train_bow)
    print("CUTOFF")
    print(f_validate_bow)
    
    y_train, xx_train = prepare_data(f_train, fit=True)
    y_vali, xx_vali = prepare_data(f_validate)
    y_test, xx_test = prepare_data(f_test) 

    return [xx_train, xx_vali, xx_test, y_train, y_vali, y_test]

'''
prepare_data taken from PS10 from CS451 S21
'''
def prepare_data(
    df: pd.DataFrame, fit: bool = False
) -> T.Tuple[np.ndarray, np.ndarray]:
    """ This function converts a dataframe to an (X, y) tuple. It learns if fit=True."""
    global numeric, scaling
    y = df.pop("immutable").values
    # use fit_transform only on training data:
    if fit:
        return y, scaling.fit_transform(numberer.fit_transform(df.to_dict("records")))
        #return y, scaling.fit_transform(df.to_dict("records"))
    # use transform on vali & test:
    return y, scaling.transform(
        numberer.transform(df.to_dict("records"))
    )  # type:ignore


if __name__ == "__main__":
    preprocessed = data_pp("full_shuffle_labeled.csv", False)
    #the below is for implementing checks of the body features generated, so far performing worse
    #preprocessed = data_pp("../bz_func_declarations/temp_final_labeled_body_shuffled.csv", True)
    features = pd.DataFrame(preprocessed)
    print(features.head())
    splits = split_data(features)
    
    x_train = splits[0]
    x_vali = splits[1]
    x_test = splits[2]
    y_train = splits[3]
    y_vali = splits[4]
    y_test = splits[5]
    #print(len(x_train))
    #print(len(y_train))
    #see if anything else needs to happen at this junctur
    #print(x_vali)
    #print(y_vali)
    print(x_vali)
    
    models = {
        "SGDClassifier": SGDClassifier(),
        "Perceptron": Perceptron(),
        "LogisticRegression": LogisticRegression()
    }
    for name, m in models.items():
        m.fit(x_train, y_train)
        print("{}:".format(name))
        print("\tVali-Acc: {:.3}".format(m.score(x_vali, y_vali)))
    
