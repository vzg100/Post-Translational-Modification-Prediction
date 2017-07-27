from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
from random import randint
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score
from imblearn.ensemble import EasyEnsemble
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import  ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

def chemical_vector(temp_window, size):
    # assumes temp_window = ProteinAnalysis(seq)
    temp_window = ProteinAnalysis(temp_window)
    return [temp_window.gravy(), temp_window.aromaticity(), temp_window.isoelectric_point()]



def generate_random_seq(locked, wing_size, center):
    amino_acids = "GALMFWKQESPVICYHRNDT"
    t1, t2 = "", ""
    for i in range(wing_size):
        t1 += amino_acids[randint(0, 19)]
        t2 += amino_acids[randint(0, 19)]
    final_seq = t1 + center + t2
    if final_seq not in locked:
        return final_seq
    else:
        generate_random_seq(locked, wing_size, center)



class DataCleaner:
    def __init__(self, file, delimit=",", header_line=0):
        self.data = pd.read_csv(file, header=header_line, delimiter=delimit, quoting=3, dtype=object)

    def load_data(self, seq="sequence", pos="position"):
        pass

    def generate_negatives(self, ratio=-1):
        pass

    def write_data(self, output):
        pass

class FastaToCSV:
    def __init__(self, fasta=[]):
        self.files = []
        for i in fasta:
            self.files.appen(open(i))

    def output(self, output):
        pass


class Pred:
    def __init__(self,  window_size=7, training_ratio=.7, seq="sequence", pos="label"):
        self.training_ratio = training_ratio  # Float value representing % of data used for training
        self.features = []
        self.labels = []
        self.window_size = window_size
        self.supervised_classifiers = {"forest": RandomForestClassifier(n_jobs=4),
                                       "mlp_adam": MLPClassifier(activation="logistic"),
                                       "svc": svm.SVC()}
        self.imbalance_functions = {"easy_ensemble": EasyEnsemble(), "SMOTEENN": SMOTEENN(),
                                    "SMOTETomek": SMOTETomek(), "ADASYN": ADASYN(),
                                    "random_under_sample": RandomUnderSampler()}
        self.seq = seq
        self.pos = pos

    def load_data(self, file, delimit=",", header_line=0):
        #Modify these if working with different CSV column names
        data = pd.read_csv(file, header=header_line, delimiter=delimit, quoting=3, dtype=object)
        self.data = data.reindex(np.random.permutation(data.index))
        for i in range(len(data[self.seq])):
            self.features.append(data[self.seq][i])
            self.labels.append(data[self.pos][i])


    def generate_random_data(self, ratio, amino_acid):
        temp_len = len(self.features)
        for i in range(int(ratio*temp_len)):
            self.features.append(generate_random_seq(center=amino_acid, wing_size=int(self.window_size*.5),
                                                     locked=self.data[self.seq]))
            self.labels.append(0)

    def vectorize(self, vectorizer):
        t = []
        for i in self.features:
            t.append(vectorizer(i, len(i)))
        self.features = t

    def balance_data(self, imbalance_function):
        imba = self.imbalance_functions[imbalance_function]
        self.features, self.labels = imba.fit_sample(self.features, self.labels)

    def supervised_training(self, classy):
        print(len(self.features), len(self.labels))
        self.classifier = self.supervised_classifiers[classy]
        temp = list(zip(self.features, self.labels))
        random.shuffle(temp)
        self.features, self.labels = zip(*temp)
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size = 0.1, random_state = 42)

        self.classifier.fit(X_train, y_train)
        test_results = self.classifier.predict(X_test)
        print("Test Results")
        print(precision_recall_fscore_support(y_test, test_results, labels=[0,1]))


    def plot(self):
        pass

x = Pred()
x.load_data(file="data/clean_s.csv")
x.generate_random_data(ratio=.2, amino_acid="S")
x.vectorize(chemical_vector)
x.supervised_training("forest")