#getting data into correct form
import csv
from typing import Dict, List, DefaultDict
from collections import defaultdict
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from sklearn.utils import resample
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from text_analysis import corp_analysis
from scipy.sparse import csr_matrix
import math
from typing import Dict, Any, List

ys = []
city_names = []
times_feats =[]
#below csv cleaning code (lines 6-21) taken from April 1 CS451 Lecture by John Foley
print("from csv to datapoint")
with open("nyt_articles_main_0.csv") as fp:
#with open("test_1_times_data.csv") as fp:
    rows = csv.reader(fp)
    header = next(rows)
    for row in rows:
        datapoint = {}
        for (column_name, column_value) in zip(header,row):
            datapoint[column_name] = column_value
        #print(datapoint)
        if datapoint['sentiment'] == 'negative':
            sent = 0
        else:
            sent = 1
        del datapoint['sentiment']
        ys.append(sent)
        string = datapoint['headline'] + " " + datapoint['abstract'] + " " + datapoint['snippet']
        '''
        del datapoint['abstract']
        del datapoint['snippet']
        del datapoint['\ufeffheadline']
        '''
        #print(string)
        inputted = corp_analysis(string)
        city_names.append([datapoint['headline'], datapoint['query']])
        #datapoint['words'] = tagged
        times_feats.append(inputted)
#need to put in dataclass?

from sklearn.base import ClassifierMixin
print("about to split up da gang")
#train/test splitting
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
#shared.py (bootstrap_auc) is taken from CS451 John Foley
from shared import bootstrap_auc, simple_boxplot
import numpy as np
#below code, lines 48-78, from Practical 08 CS451 by John Foley
RANDOM_SEED = 1234

print('city names len: ' + str(len(city_names)))
#turning shuffle off right now
# split off train/validate (tv) pieces.
ex_tv, ex_test, y_tv, y_test = train_test_split(
    times_feats,
    ys,
    train_size=0.10,
    shuffle=False,
    random_state=RANDOM_SEED,
)

xx_tv, xx_test, yx_tv, yx_test = train_test_split(
    city_names,
    ys,
    train_size=0.10,
    shuffle=False,
    random_state=RANDOM_SEED

)
print('ex_test len: ' + str(len(ex_test)))
print('xx_tv len: ' + str(len(xx_tv)))
# split off train, validate from (tv) pieces.
ex_train, ex_vali, y_train, y_vali = train_test_split(
    ex_tv, y_tv, train_size=0.9, shuffle=False, random_state=RANDOM_SEED
)

xx_train, xx_vali, yx_train, yx_vali = train_test_split(
    xx_tv, yx_tv, train_size=0.9, shuffle=False, random_state=RANDOM_SEED
)

#turn into bag-of-words format (portions of this taken from Practial 06, CS451, by John Foley)

print("breaking down text")


#this is CountVectorization
from sklearn.feature_extraction.text import CountVectorizer
#add in nltk
'''
word_features = CountVectorizer(
    strip_accents="unicode",
    lowercase=False,
    ngram_range=(1, 3),
)

#word_features = TfidfVectorizer()

text_to_words = word_features.build_analyzer()
#ex_train needs to be in string format, not dict
word_features.fit(ex_train)

X_train = word_features.transform(ex_train)
X_vali = word_features.transform(ex_vali)
X_test = word_features.transform(ex_test)

'''

#using TfidVectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Only learn columns for words in the training data, to be fair.
word_to_column = TfidfVectorizer(
    strip_accents="unicode", lowercase=True, stop_words="english", max_df=0.5
)
word_to_column.fit(ex_train)

# Test words should surprise us, actually!
X_train = word_to_column.transform(ex_train)
X_vali = word_to_column.transform(ex_vali)
X_test = word_to_column.transform(ex_test)


print(X_train.shape, X_vali.shape, X_test.shape)

#results: Dict[str, List[float]] = {}

print("going into learning")
#models 
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
#also do kNN classifier

from sklearn.pipeline import make_pipeline
from sklearn.svm import NuSVC

#LinearModel taken from Practical 07 CS451, John Foley
@dataclass
class LinearModel:
    weights: np.ndarray  # note we can't specify this is 1-dimensional
    bias: float = 0.0

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """ Compute the signed distance from the self.weights hyperplane. """
        (N, D) = X.shape
        assert self.weights.shape == (D, 1)
        # Matrix multiplication; sprinkle transpose and assert to get the shapes you want (or remember Linear Algebra)... or both!
        output = np.dot(self.weights.transpose(), X.transpose())
        assert output.shape == (1, N)
        return (output + self.bias).reshape((N,))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Take whether the points are above or below our hyperplane as a prediction. """
        return self.decision_function(X) > 0

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """ Take predictions and compute accuracy. """
        y_hat = self.predict(X)
        return metrics.accuracy_score(np.asarray(y), y_hat)

    def compute_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        """ Distance to hyperplane is used for AUC-style metrics. """
        return metrics.roc_auc_score(y, self.decision_function(X))



#ModelTrainingCurve taken from Practical 07 CS451, John Foley
@dataclass
class ModelTrainingCurve:
    train: List[float] = field(default_factory=list)
    validation: List[float] = field(default_factory=list)

    def add_sample(
        self,
        m: LinearModel,
        X: np.ndarray,
        y: np.ndarray,
        X_vali: np.ndarray,
        y_vali: np.ndarray,
    ) -> None:
        self.train.append(m.score(X, y))
        self.validation.append(m.score(X_vali, y_vali))

learning_curves: DefaultDict[str, ModelTrainingCurve] = defaultdict(ModelTrainingCurve)
#MLP Classifier
'''
params = {
    "hidden_layer_sizes": (32,),
    "random_state": 5,
    "max_iter": 500,
    "alpha": 0.0001,
}
mlp = MLPClassifier(**params)
for iter in tqdm(range(400)):
    for iter_1 in range(10):
        mlp.partial_fit(X_train, y_train, classes=(0, 1))
        learning_curves["mlpPercep"].add_sample(
            mlp, X_train, y_train, X_vali, y_vali)
print("mlp. Train-Accuracy: {:.3}".format(mlp.score(X_train, y_train)))
print("mlp. Vali-Accuracy: {:.3}".format(mlp.score(X_vali, y_vali)))
'''
'''
#checking if more labels will help
N = len(y_train)
num_trials = 100
percentages = list(range(5, 100, 5))
percentages.append(100)
scores = {}
acc_mean = []
acc_std = []

# Which subset of data will potentially really matter.
for train_percent in percentages:
    n_samples = int((train_percent / 100) * N)
    print("{}% == {} samples...".format(train_percent, n_samples))
    label = "{}".format(train_percent, n_samples)

    # So we consider num_trials=100 subsamples, and train a model on each.
    scores[label] = []
    for i in range(num_trials):
        X_sample, y_sample = resample(
            X_train, y_train, n_samples=n_samples, replace=False
        )  # type:ignore
        # Note here, I'm using a simple classifier for speed, rather than the best.
        clf = SGDClassifier(random_state=RANDOM_SEED + train_percent + i)
        clf.fit(X_sample, y_sample)
        # so we get 100 scores per percentage-point.
        scores[label].append(clf.score(X_vali, y_vali))
    # We'll first look at a line-plot of the mean:
    acc_mean.append(np.mean(scores[label]))
    acc_std.append(np.std(scores[label]))

# First, try a line plot, with shaded variance regions:
import matplotlib.pyplot as plt

means = np.array(acc_mean)
std = np.array(acc_std)
plt.plot(percentages, acc_mean, "o-")
plt.fill_between(percentages, means - std, means + std, alpha=0.2)
plt.xlabel("Percent Training Data")
plt.ylabel("Mean Accuracy")
plt.xlim([0, 100])
plt.title("Shaded Accuracy Plot")
#plt.savefig("graphs/p09-area-Accuracy.png")
plt.show()


# Second look at the boxplots in-order: (I like this better, IMO)
simple_boxplot(
    scores,
    "Learning Curve",
    xlabel="Percent Training Data",
    ylabel="Accuracy",
    save="graphs/p09-boxplots-Accuracy.png",
)

'''

#%% Define & Run Experiments
@dataclass
class ExperimentResult:
    vali_acc: float
    params: Dict[str, Any]
    model: ClassifierMixin
'''
#decision tree
print("Consider Decision Tree.")
performances: List[ExperimentResult] = []

for rnd in range(3):
    for crit in ["entropy"]:
        for d in range(1, 9):
            params = {
                "criterion": crit,
                "max_depth": d,
                "random_state": rnd,
            }
            f = DecisionTreeClassifier(**params)
            f.fit(X_train, y_train)
            vali_acc = f.score(X_vali, y_vali)
            result = ExperimentResult(vali_acc, params, f)
            performances.append(result)
best_dtree = max(performances, key=lambda result: result.vali_acc)
print("best dtree: ", best_dtree)
'''
'''
print("kNN results below")
for neigh in tqdm(range(20)):
    m = KNeighborsRegressor(n_neighbors=5, weights="distance")
    m.fit(X_train, y_train)
    print("k: " + str(neigh) + ", acc: " + str(m.score(X_vali, y_vali)))
'''
'''
print("Consider Logistic Regression.")
performances_lr: List[ExperimentResult] = []
for rnd in range(3):
    params = {
        "random_state": rnd,
        "penalty": "l2",
        "max_iter": 100,
        "C": 1.0,
    }
    f = LogisticRegression(**params)
    f.fit(X_train, y_train)
    vali_acc = f.score(X_vali, y_vali)
    result = ExperimentResult(vali_acc, params, f)
    performances_lr.append(result)

best_lr = max(performances_lr, key=lambda result: result.vali_acc)
print("best log_reg: ", best_lr)
'''
'''

# Note that Sci-Kit Learn's Perceptron uses an alternative method of training.
# Is it an averaged perceptron or a regular perceptron?
skP = Perceptron()
print("Train sklearn-Perceptron (skP)")
for iter in tqdm(range(1000)):
    # Note we use partial_fit rather than fit to expose the loop to our code!
    skP.partial_fit(X_train, y_train, classes=(0, 1))
    learning_curves["skPerceptron"].add_sample(skP, X_train, y_train, X_vali, y_vali)
print("skP. Train-Accuracy: {:.3}".format(skP.score(X_train, y_train)))
print("skP. Vali-Accuracy: {:.3}".format(skP.score(X_vali, y_vali)))

'''
from sklearn.svm import SVC as SVMClassifier
#first model, MultinomalNB
results: Dict[str, List[float]] = {}
#svc
'''
training accuracy: 0.9980676328502416
validation accuracy: 0.75
'''
results = {}
#clf = make_pipeline(StandardScaler(with_mean=False), NuSVC())
Cs = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
for class_weights in [None, "balanced"]:
    for yamma in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        for value in tqdm(Cs):
            clf = SVMClassifier(C=value, class_weight=class_weights, gamma=yamma, kernel='rbf' )
            clf.fit(X_train, y_train)
            train_scores = clf.score(X_train, y_train)
            vali_scores = clf.score(X_vali, y_vali)
            print('settings are, rbf, C = {} {} gamma = {}'.format(
                    value, class_weights or '', yamma
                ))
            print("training accuracy: " + str(train_scores))
            print("validation accuracy: " + str(vali_scores))
            results[vali_scores] = [value, class_weights or '', yamma, vali_scores, train_scores]

print(results[max(results.keys())])

'''
final_product = []

#print('x_validation len: ' + str(len(X_vali)))
for alpha in [0.5]:
    m = MultinomialNB(alpha=alpha)
    m.fit(X_train, y_train)
    
    predictions = m.predict(X_test)
    print(len(predictions))
    #scores = m.predict_proba(X_test)[:, 1]
    print(len(xx_test))
    #print(X_vali[0])
    
    for index in range(len(xx_test)):
        final_product.append([xx_test[index][1], xx_test[index][0], predictions[index]])
    print("final_product len: " + str(len(final_product)))
    
    print(
        "Accuracy: {:.3}, AUC: {:.3}".format(
            m.score(X_vali, y_vali), roc_auc_score(y_score=scores, y_true=y_vali)
        )
    )
    '''
    #print("What I called log(beta)={}".format(m.class_log_prior_[1]))
'''results["MNB(alpha={})".format(alpha)] = bootstrap_auc(m, X_vali, y_vali)'''
    #learning_curves["MNB(alpha={})".format(alpha)].add_sample(m, X_train, y_train, X_vali, y_vali)
'''

for entry in final_product:
        print(entry)

with open("final_exploration_data.csv", 'w', newline='') as fp_0:
    writer = csv.writer(fp_0)
    writer.writerow(['city', 'headline', 'score'])
    for entry in final_product:
        writer.writerow(entry)

'''
'''
simple_boxplot(results, ylabel="AUC")

for key, dataset in learning_curves.items():
    xs = np.array(list(range(len(dataset.train))))
    # line-plot:
    plt.plot(xs, dataset.train, label="{} Train".format(key), alpha=0.7)
    plt.plot(xs, dataset.validation, label="{} Validate".format(key), alpha=0.7)
    # scatter-plot: (maybe these look nicer to you?)
    # plt.scatter(xs, points, label=key, alpha=0.7, marker=".")
    plt.ylim((0.4, 1.1))
    plt.title("{} Learning Curves [norm={}]".format(key, "none"))
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    #plt.savefig("graphs/p07-{}-curve-{}.png".format(key, norm))
    plt.show()
'''

