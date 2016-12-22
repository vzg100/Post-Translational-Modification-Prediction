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
    prefeatures, prelabels, labels, features = [], [], [], []

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
                self.prefeatures.append([[self.hydrophobicity[amino_acid] for amino_acid in seq],
                                         [self.electronegativity[amino_acid] for amino_acid in seq],
                                         [1 if amino_acid in self.phosphorylations_sites else 0 for amino_acid in seq],
                                         [i/len(seq) for i in range(len(seq))]])
                self.prelabels.append([1 if residue == (sline[-1]) else 0 for residue in range(len(sline[4]))])

            else:
                print("Skipped Header")
            counter += 1

    def sklearn_jazz(self, splitratio):
        # handle splitting first
        # Worry about matrix encoding
        splitter = int(splitratio*len(self.prefeatures))
        training_features = np.array(self.prefeatures[splitter:], dtype=object)
        training_labels = self.prelabels[splitter:]
        testing_features = self.prefeatures[:splitter]
        testing_lables = self.prelabels[:splitter]
        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(training_features, training_labels)
x = ProteinLearner("temp_test.csv", 99)
zz = x.prefeatures
x.sklearn_jazz(.8)
