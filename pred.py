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


def distance(s1: str, s2: str, threshold:float =.9):
    t = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            t+=1
    if (len(s1)-t)/s2 < threshold:
        return False
    else:
        return True


def windower(sequence: str, position: int, wing_size: int):
    # window size = wing_size*2 +1
    position = int(position)
    wing_size = int(wing_size)
    if (position - wing_size) < 0:
        return sequence[:wing_size + position]
    if (position + wing_size) > len(sequence):
        return sequence[position - wing_size:]
    else:
        return sequence[position - wing_size:position + wing_size]


def chemical_vector(temp_window: str):
    temp_window = ProteinAnalysis(temp_window)
    return [temp_window.gravy(), temp_window.aromaticity(), temp_window.isoelectric_point()]


def generate_random_seq(locked: list, wing_size: int, center: str):
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
    def __init__(self, file: str, delimit: str =",", header_line: int=0, wing=10):
        self.data = pd.read_csv(file, header=header_line, delimiter=delimit, quoting=3, dtype=object)
        self.protiens = {}
        self.count = 0
        self.labels = []
        self.sequences = []
        self.wing = wing

    def load_data(self, seq: str="sequence", pos:str ="position"):
        for i in range(len(self.data[seq])):
            t = self.data[seq][i]
            if t not in self.protiens.keys():
                self.protiens[t] = [self.data[pos][i]]
            else:
                self.protiens[t] = self.protiens[t].append(self.data[pos][i])

    def generate_positive(self):
        for i in self.protiens.keys():
            t = self.protiens[i]
            for j in t:
                self.sequences.append(windower(sequence=i, position=j-1, wing_size=self.wing))
                self.labels.append(1)

    def generate_negatives(self, amino_acid: str, ratio: int=-1, cross_check: int=-1):
        self.count = len(list(self.protiens.keys()))
        if ratio < 0:
            for i in self.protiens.keys():
                for j in range(len(i)):
                    if i[j] == amino_acid and j+1 not in self.protiens[i]:
                        self.sequences.append(windower(sequence=i, position=j, wing_size=self.wing))
                        self.labels.append(0)
        else:
            t = len(self.sequences)
            for y in range(int(t*ratio)):
                for i in self.protiens.keys():
                    for j in range(len(i)):
                        if i[j] == amino_acid and j + 1 not in self.protiens[i]:
                            s = windower(sequence=i, position=j, wing_size=self.wing)
                            if cross_check < 0:
                                self.sequences.append(s)
                                self.labels.append(0)
                            else:
                                for subsequence in self.sequences:
                                    if not distance(s1=subsequence, s2=s):
                                        self.sequences.append(s)
                                        self.labels.append(0)


    def write_data(self, output: str, seq_col: str, label_col: str, shuffle = 0):
        file = open(output, "W")
        t = str(seq_col) + "," + str(label_col)
        file.write(t)
        if shuffle != 0:
            temp = list(zip(self.sequences, self.labels))
            random.shuffle(temp)
            self.sequences, self.labels = zip(*temp)
        for i in range(len(self.sequences)):
            file.write(str(self.sequences[i]) + "," + self.labels[i])


class FastaToCSV:
    # More Benchmarks
    def __init__(self, fasta: list):
        self.files = []
        for i in fasta:
            self.files.append(open(i))

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
        # Modify these if working with different CSV column names
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
            t.append(vectorizer(i))
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
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels,
                                                            test_size = 0.1, random_state = 42)

        self.classifier.fit(X_train, y_train)
        test_results = self.classifier.predict(X_test)
        print("Test Results")
        print(precision_recall_fscore_support(y_test, test_results, labels=[0, 1]))


    def plot(self):
        pass

#Remove Randoms from Test Data set

x = Pred()
x.load_data(file="data/clean_s.csv")
x.generate_random_data(ratio=.2, amino_acid="S")
x.vectorize(chemical_vector)
x.supervised_training("forest")
