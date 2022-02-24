import csv
import ntlk
import math
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import resample
from collections import defaultdict
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler


RANDOM_SEED = 1999
#hypothesis: while the model will be more efficient than other approaches, it will have difficulty labeling in such a constrained context

def data_pp(data):



def split_data(xs, ys):
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



