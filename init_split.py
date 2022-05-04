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
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_regression
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datetime import datetime
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
#import matplotlib as plt
from text_analysis import lemmatize_bagged, main_analysis, get_return_value, get_param_location, param_traits, action_on_sub
from body_analysis import body_len, find_left_invoke

ys = []
prototypes = []
RANDOM_SEED = 1999
#numberer = DictVectorizer(sparse=True)
numberer = FeatureHasher()
scaling = StandardScaler(with_mean=False)
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
                #print(new_datapoint["body_length"])
            just_words = {}
            just_words["immutable"] = immutable
            for word in new_datapoint["lemmatized_bow"]:
                    new_datapoint["func_dec: " + word] = 1
                    just_words["func_dec: " + word] = 1
                    #print(word)
            #new_datapoint["relevant_action"] = action_on_sub(datapoint["func_prototype"], target_param)
            #adding to big set
            new_datapoint['relevant_action_sub'] = action_on_sub(datapoint["func_prototype"], target_param)
            prototypes.append(just_words)
            #for testing 
            location += 1
            print(location)
            if location == 20000:
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
    '''
    func_to_column = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, stop_words="english", max_df=0.5
    )

    func_to_column.fit(f_train["func_prototype"].to_list())
    
    different_data_sets =  [f_train, f_validate, f_test]
    
    for dataset in different_data_sets:
        for bag in dataset["lemmatized_bow"]:
            #dataset["funcdec: " + word] = 1
            for word in bag:
                dataset["func_dec: " + word] = 1
    

    f_train_bow = func_to_column.transform(f_train["func_prototype"].to_list())
    f_validate_bow = func_to_column.transform(f_validate["func_prototype"].to_list())
    f_test_bow = func_to_column.transform(f_test["func_prototype"].to_list())
    '''

    #del f_train["lemmatized_bow"]
    #del f_validate["lemmatized_bow"]
    #del f_test["lemmatized_bow"]
    #f_train["func_col"] = f_train_bow
    y_train_temp = f_train.pop("immutable").values
    print("testing")
    get_coeff(f_train, y_train_temp)
    f_train["immutable"] = y_train_temp
    #pd.options.display.max_colwidth = 200
    #print(f_train_bow)
    #print("CUTOFF")
    #print(f_validate_bow)
    
    y_train, xx_train = prepare_data(f_train, fit=True)
    y_vali, xx_vali = prepare_data(f_validate)
    y_test, xx_test = prepare_data(f_test) 

    return [xx_train, xx_vali, xx_test, y_train, y_vali, y_test]

def get_coeff(x_train, y_train):
    print(type(x_train))
    print(len(y_train))
    model =  LogisticRegression(max_iter=10000, random_state=1841, solver='sag')
    model.fit(x_train, y_train)
    coefs = pd.DataFrame( model.coef_[0],columns=['Coefficients'], index=x_train.columns)
    coefs = coefs.sort_values(by = ["Coefficients"], ascending=False)
    print(coefs.head)
    with open("getting_test_coefficients_test.txt", "w") as f:
        f.write(str(coefs.head))
    coefs.to_csv("ranked_coefs_LogReg.csv")

'''
prepare_data taken from PS10 from CS451 S21
'''
def prepare_data(
    df: pd.DataFrame, fit: bool = False
) -> T.Tuple[np.ndarray, np.ndarray]:
    """ This function converts a dataframe to an (X, y) tuple. It learns if fit=True."""
    global numeric, scaling
    y = df.pop("immutable").values
    #return y, df.to_dict('records')
    # use fit_transform only on training data:
    if fit:
        return y, scaling.fit_transform(numberer.fit_transform(df.to_dict("records")))
        #return y, scaling.fit_transform(df.to_dict("records"))
    # use transform on vali & test:
    return y, scaling.transform(
        numberer.transform(df.to_dict("records"))
    )  # type:ignore


if __name__ == "__main__":
    loading_data_in = datetime.now()
    #preprocessed = data_pp("../various_data/full_shuffle_labeled.csv", False)
    #the below is for implementing checks of the body features generated, so far performing worse
    preprocessed = data_pp("../temp_final_labeled_body_shuffled.csv", True)
    features = pd.DataFrame(preprocessed)
    
    with open("getting_test_.txt", "w") as f: 
    #this is where to take out handcrafted
        hand_crafted_features = ["func_return_value", "parameter_location", "param_name_len", "relevant_action_sub", "body_length", "left_of_eq", "label_time"]
        for removed_param in hand_crafted_features:
            taken_out = features.loc[:,removed_param]
            #features.drop(removed_param, inplace=True, axis=1)
    
            #get the pd.df to replace NaN
            print(features.head())
            #features.replace(NaN, 0)
            features.fillna(0, inplace=True)
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
            #rint(x_vali)
            #print(y_vali)
            #print(x_train.shape)
            #test this below line with chi2/mutual_info_regression
            
            '''
            feature_op = SelectPercentile(f_classif, percentile=85)
            x_train_new = feature_op.fit_transform(x_train, y_train)
            print(x_vali)
            x_vali_new = feature_op.transform(x_vali)
            x_test_new = feature_op.transform(x_test)
            #doing hyperparam optimization here
        
            param_grid = [{'alpha': [0.1, 0.01, 0.001, 0.5], 'max_iter': [1500, 2000, 1000], 'random_state':[1841]}]
            base_estimator = Perceptron()
            sh = GridSearchCV(base_estimator, param_grid).fit(x_train_new, y_train)
            print(sh.best_estimator_)
            df = pd.DataFrame(sh.cv_results_)
            print(df.head())
            '''
            models = {
                "SGDClassifier": SGDClassifier(),
                "Perceptron": Perceptron(alpha=0.1, max_iter=1500, random_state=1841),
                "LogisticRegression": LogisticRegression(max_iter=10000, random_state=1841, solver='sag')
            }
            prep_time = datetime.now() - loading_data_in
            f.write("prep time: " + str(prep_time))
            for name, m in models.items():
                start_model = datetime.now()
                m.fit(x_train, y_train)
                #try to plot the training curve at this moment?
                print("{}:".format(name))
                f.write(name)
                vali_acc = m.score(x_vali, y_vali)
                print("\tVali-Acc: {:.3}".format(vali_acc))
                y_predictions = m.predict(x_test)
                if name != "SGDClassifier" and name != "Perceptron":
                    y_probs = m.predict_proba(x_test)
                    #names= m.feature_names_in_[0]
                    #print(names)
                    #feature_importance = pd.DataFrame(list(features.columns), columns = ["feature"])
                    importance = m.coef_[0]
                    #feature_importance["importance"] = pow(math.e, importance)
                    #feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)
                    #print(feature_importance.head)
                    columns = list(x_train.columns)
                    print(columns)
                    for i,v in enumerate(importance):
                        if v != 0:
                            #print(columns[i])
                            print('Feature: %0d, Score: %.5f' % (i,v))
                    #for i,v in enumerate(importance):
                    #    print('Feature: ' + columns[i] + ', Score: '+  str(v))
                #prec_recall_array = precision_recall_fscore_support(y_test, y_predictions, average='macro')
                #precision, recall, _ = precision_recall_curve(y_test, y_predictions)
                done_model = datetime.now() - start_model
                #pr_plot = PrecisionRecallDisplay(y_test, y_predictions)
                #plt.savefig('foo' + name + '.png')

                if name != "SGDClassifier"and name != "Perceptron":
                    temp_df = pd.DataFrame({'predictions': y_predictions, 'truth':y_test})
                    prob_df = pd.DataFrame(y_probs)
                    prob_df.to_csv("logreg_prob_test_4_26.csv")
                else:
                    temp_df = pd.DataFrame({'predictions': y_predictions, 'truth':y_test})
                temp_df.to_csv('test_results_' + name + '_4_26_again.csv')
                test_acc = accuracy_score(y_test, y_predictions)
                '''
                https://datascience.stackexchange.com/questions/81389/plotting-multiple-precision-recall-curves-in-one-plot
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay
                PrecisionRecallDisplay (need to do with multiple models?)
                temp_df = pd.DataFrame({'predictions': y_predictions, 'probabilities': y_probs, 'truth':y_test})
                temp_df.to_csv('test_results_' + name + '.csv')
                record the other pieces of information in a txt file
                    - time
                    - accuracy
                    - 

                '''
                f.write(removed_param + "test: "+ str(test_acc) + "; time: " + str(done_model))
                f.write("------------") 
            break
        #features[removed_param] = taken_out
