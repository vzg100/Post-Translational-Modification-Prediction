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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


def windower(sequence, position, wing_size):
    # window size = wing_size*2 +1
    position = int(position)
    wing_size = int(wing_size)
    if (position - wing_size) < 0:
        return sequence[:wing_size + position]
    if (position + wing_size) > len(sequence):
        return sequence[position - wing_size:]
    else:
        return sequence[position - wing_size:position + wing_size]


def chemical_vector(temp_window, size):
    # assumes temp_window = ProteinAnalysis(seq)
    temp_window = ProteinAnalysis(temp_window)
    q = [temp_window.gravy(),
         temp_window.aromaticity(),
         temp_window.isoelectric_point()
         ]

    return q


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


def report_results(results, answers, classy, shift=0):
    # Turn this into a class
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(results)):
        if results[i] == 1 and answers[i+shift] == 1:
            tp += 1
        elif results[i] == 0 and answers[i+shift] == 0:
            tn += 1
        elif results[i] == 1 and answers[i+shift] == 0:
            fp += 1
        else:
            fn += 1
    if tp != 0 and tn != 0:
        tpr = tp / (tp + fn)  # aka recall aka true positive rate
        spc = tn / (tn+fp)  # specificty or true negative rate
        ppv = tp / (tp + fp)  # positive predicative value aka precision
        npv = tn/(tn+fn)  # negative predictive value
        fpr = fp/(fp+tn)  # false positive rate aka fallout
        fnr = fn/(tp+fn)  # false negative rate
        fdr = fp/(tp+fp)  # false discovery rate
        acc = (tp + tn) / (tp + fp + tn + fn)
        roc = roc_auc_score(answers, results)
        inf = (tpr+spc)-1
        mkd = (ppv+npv)-1
        print(classy)
        print("Sensitivity:"+str(tpr))
        print("Specificity :" + str(spc))
        print("Positive Predictive Value:" + str(ppv))
        print("Negative Predictive Value:" + str(npv))
        print("False Positive Rate:" + str(fpr))
        print("False Negative Rate:" + str(fnr))
        print("False Discovery Rate:" + str(fdr))
        print("Accuracy:" + str(acc))
        print("ROC:" + str(roc))

        print("\n\n")
        return [tpr, spc, ppv, npv, fpr, fnr, fdr, acc, inf, mkd]
    else:
        print("Failed")
        return False


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
        self.classifier = self.supervised_classifiers[classy]





x = Pred()
x.load_data(file="data/clean_s.csv")
x.generate_random_data(.2, "K")
x.vectorize(chemical_vector)
x.balance_data("ADASYN")