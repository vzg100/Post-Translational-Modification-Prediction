from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy as np

class ProteinLearner:
    # Conversion Keys
    # pulled from http://www.sigmaaldrich.com/life-science/metabolomics/learning-center/amino-acid-reference-chart.html
    hydrophobicity = {"l": 97, "i": 99, "f": 100, "w": 97, "v": 76, "m":74, "y": 63, "c": 49, "a": 41, "t": 13, "h": 8,
                      "g": 0, "s": -5, "q": -10, "r": -14, "k": -23, "n": -28, "e": -31, "p": -46, "d": -55}
    phosphorylations_sites = ["r", "s", "w"]
    electronegativity = {"l": 0, "i": 0, "f": 0, "w": 0, "v": 0, "m":0, "y": 0, "c": -1, "a": 0, "t": -1, "h": 2,
                         "g": 0.5, "s": -1, "q": 1, "r": 2, "k": 2, "n": 1, "e": -2, "p": 1, "d": -2}
    # Properties -> gonna be stored as ints or bits
    sequences, prefeatures, prelabels, labels, features = [], [], [], [], []

    def __init__(self, protfile, testsize):
        f = open(protfile, 'r')

        counter = 0
        for line in f:

            if counter == testsize:
                break
            if counter != 0:
                sline = line.split(",")
            # Residue, ID, Seq, Res Position
                seq = sline[4].lower()
                self.sequences.append([i for i in line[2]])
            else:
                print("Skipped Header")
            counter += 1

