import csv
import nltk
import math
#import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import resample
from collections import defaultdict
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

ys = []
prototypes = []
RANDOM_SEED = 1999
#hypothesis: while the model will be more efficient than other approaches, it will have difficulty labeling in such a constrained context

def data_pp(data):
    with open(data) as fp:
        location = 0
        rows = csv.reader(fp)
        header = next(rows)
        for row in rows:
            datapoint = {}
            for (column_name, column_value) in zip(header,row):
                datapoint[column_name] = column_value
            print(datapoint)
            #formatting data point binary in 0/1
            if datapoint['immutable'] == False:
                immutable = 0
            else:
                immutable = 1
            del datapoint['immutable']
            ys.append(immutable)
            target_param = ""
            #finding the targeted param
            for i in range(10):
                current_param_entry = datapoint["a" + str(i)]
                if current_param_entry != "ignore" and current_param_entry != "u":
                    target_param = current_param_entry
                    break
            #print(target_param)
            prototypes.append([datapoint["func_prototype"], target_param])
            
            location += 1
            if location == 3:
                break
        



def split_data(xs, ys):
    #if generating more features, may need multiple splitting things 
    xx_tv, xx_test, yx_tv, yx_test = train_test_split(
        xs,
        ys,
        train_size=0.80,
        shuffle=True,
        random_state=RANDOM_SEED
    )
    xx_train, xx_vali, yx_train, yx_vali = train_test_split(
        xx_tv, yx_tv, train_size=0.8, shuffle=False, random_state=RANDOM_SEED
    )


#TODO: generate a function that can generate more robust features


if __name__ == "__main__":
    data_pp("full_shuffle_labeled.csv")


