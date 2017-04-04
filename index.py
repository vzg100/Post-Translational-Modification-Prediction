from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
from random import randint
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn import svm


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


def featurify(temp_window):
    # assumes temp_window = ProteinAnalysis(seq)
    q = [temp_window.gravy(),temp_window.aromaticity(), temp_window.isoelectric_point(),temp_window.instability_index(), 
        temp_window.secondary_structure_fraction()[0],temp_window.secondary_structure_fraction()[1],
        temp_window.secondary_structure_fraction()[2]]
    z = temp_window.amino_acids_content

    for i in "GALMFWKQESPVICYHRGNDT":
        q.append(z[i])
    q = q

    return q

def random_seq(locked, wing_size, center):
    amino_acids = "GALMFWKQESPVICYHRGNDT"
    t1, t2 = "", ""
    for i in range(wing_size):
        t1 += amino_acids[randint(0, 20)]
        t2 += amino_acids[randint(0, 20)]
    final_seq = t1 + center + t2
    if final_seq not in locked:
        return final_seq
    else:
        random_seq(locked, wing_size, center)


def report(results, answers, shift=0):
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
    if tp != 0:
        tpr = tp / (tp + fn)  #aka recall aka true positive rate
        spc =tn / (tn+fp)  #specificty or true negative rate
        ppv = tp / (tp + fp)  # positive predicative value aka precision
        npv = tn/(tn+fn)  #negative predictive value
        fpr = fp/(fp+tn)  # false positive rate aka fallout
        fnr = fn/(tp+fn)  #false negative rate
        fdr = fp/(tp+fp)  # false discovery rate
        acc = (tp + tn) / (tp + fp + tn + fn)
        #roc = roc_auc_score(answers, results[shift:])
        inf = (tpr+spc)-1
        mkd = (ppv+npv)-1
        print("Sensitivity:"+str(tpr))
        print("Specificity :" + str(spc))
        print("Positive Predictive Value:" + str(ppv))
        print("Negative Predictive Value:" + str(npv))
        print("False Positive Rate:" + str(fpr))
        print("False Negative Rate:" + str(fnr))
        print("False Discovery Rate:" + str(fdr))
        print("Accuracy:" + str(acc))
        #print("ROC:" + str(roc))
        print("")
        return [tpr, spc, ppv, npv, fpr, fnr, fdr, acc, inf, mkd]
    else:
        print("Failed")
        return False

class Classy:

    def __init__(self, data="phosphosites.csv", delimit=",", amino_acid="Y", sites="code",
                 modification="phosphorylation", window_size=7, pos="position", training_ratio=.7,
                 header_line=0, seq="sequence", neg_per_seq=5, lines_to_read=90000, forest_size=110, classy="forest"):
        self.classy = classy
        data = pd.read_csv(data, header=header_line, delimiter=delimit, quoting=3, dtype=object)
        self.data = data.reindex(np.random.permutation(data.index))
        self.amino_acid = amino_acid
        self.training_ratio = training_ratio  # Float value representing % of data used for training
        self.proteins = {}
        self.neg_count = 0
        self.neg_per_seq = neg_per_seq
        self.window = int(window_size)
        self.features= []
        self.labels= []
        self.pos_features= []
        self.neg_features= []
        self.pos_seq= []

        for i in range(lines_to_read):
            if ("X" not in data[seq][i]) and (data[sites][i] == amino_acid) and (data[seq][i] not in self.proteins.keys()):
                self.proteins[data[seq][i]] = [data[pos][i]]
            elif ("X" not in data[seq][i]) and (data[sites][i] == amino_acid) and (
                data[pos][i] not in self.proteins[data[seq][i]]):
                self.proteins[data[seq][i]].append(data[pos][i])

        for i in self.proteins.keys():
            neg_sites = []
            for position in self.proteins[i]:
                temp_window = ProteinAnalysis(windower(i, position, self.window))
                self.pos_seq.append(windower(i, position, self.window))
                self.pos_features.append(featurify(temp_window))
            for amino_acid_sites in range(len(i)):
                # creates list of potential negative sites from the current sequence
                if i[amino_acid_sites] == self.amino_acid and amino_acid_sites not in self.proteins[i]:
                    neg_sites.append(amino_acid_sites)
            counter = 0
            neg_sites_used = []
            while (counter < self.neg_per_seq) and (len(neg_sites_used) != len(neg_sites)):
                temp_negative = randint(0, len(neg_sites))
                if temp_negative not in neg_sites_used:
                    temp_window = ProteinAnalysis(windower(i, temp_negative, self.window))
                    counter += 1
                    self.neg_features.append(featurify(temp_window))
                    self.neg_count +=1

    def generate_data(self, random_=1, random_ratio=2, random_test=0):
        rand_features = []
        neg_labels = [0 for i in range(len(self.neg_features))]
        pos_labels = [1 for i in range(len(self.pos_features))]
        if random_ == 1 and random_ratio > 0:
            for i in range(int((len(self.pos_features)+len(self.neg_features))*random_ratio)):
                rand_features.append(featurify(ProteinAnalysis(random_seq(locked=self.pos_seq, wing_size=self.window, center=self.amino_acid))))
        if random_test == 0:
            features = self.pos_features+self.neg_features
            labels = pos_labels+neg_labels
            temp = list(zip(features, labels))
            random.shuffle(temp)
            features, labels = zip(*temp)
            training_slice = int(self.training_ratio * len(labels))
            self.training_features = list(features[:training_slice])+rand_features
            self.training_labels = list(labels[:training_slice])+[0 for i in range(len(rand_features))]
            self.test_features = features[training_slice:]
            self.test_labels = labels[training_slice:]
        else:
            features = self.pos_features+self.neg_features+rand_features
            labels = pos_labels+neg_labels+[0 for i in range(len(rand_features))]
            temp = list(zip(features, labels))
            random.shuffle(temp)
            features, labels = zip(*temp)
            training_slice = int(self.training_ratio * len(labels))
            self.training_features = list(features[:training_slice])
            self.training_labels = list(labels[:training_slice])
            self.test_features = features[training_slice:]
            self.test_labels = labels[training_slice:]

    def calculate(self):
        classif = {"forest": RandomForestClassifier(verbose=0, n_jobs=4),
                           "mlp_adam": MLPClassifier(solver='adam', random_state=1),
                           "svc": svm.SVC(), "l_svc": svm.LinearSVC(),
                           "mlp_sgd": MLPClassifier(solver='sgd', random_state=1),
                           "mlp_lbfgs": MLPClassifier(solver='lbfgs', random_state=1)
                           }
        self.clf = classif[self.classy]
        self.clf.fit(self.training_features, self.training_labels)
        self.results = self.clf.predict(self.test_features)
        self.rating = precision_recall_fscore_support(self.test_labels, self.results,average="macro")
    def speak_to_the_trees(self):
        feat_imp = self.clf.feature_importances_
        print(feat_imp)
        return feat_imp

    def report(self):
        print("Report for this run\n" +
          "Amino Acid: " + self.amino_acid + "\n" +
          " Classy: " + self.classy + "\n" +
          " Pos-Neg: " + str(self.neg_per_seq) + "\n" +
          " window_size: " + str(self.window) + "\n")
        report(answers=self.test_labels, results = self.results)

    def test(self, positive_file, negative_file, sequence_position=10):
        # for my test files sequence position = 10
        test_features = []
        test_labels = []
        with open(positive_file) as f:
            for i in f:
                if ">" not in i and i[sequence_position] == self.amino_acid:
                    temp_window = ProteinAnalysis(windower(i, sequence_position, self.window).strip("\t"))
                    feat = featurify(temp_window)
                    test_features.append(feat)
                    test_labels.append(1)
        with open(negative_file) as f:
            for i in f:
                if ">" not in i and i[sequence_position] == self.amino_acid:
                    temp_window = ProteinAnalysis(windower(i, sequence_position, self.window).strip("\t"))
                    feat = featurify(temp_window)
                    test_features.append(feat)
                    test_labels.append(0)
        temp = list(zip(test_features, test_labels))
        random.shuffle(temp)
        test_features, test_labels = zip(*temp)
        test_results = self.clf.predict(test_features)
        report(test_results, test_labels)

for site in "SYTK":
    for classy in ["mlp_adam", "svc", "forest"]:
        for ratio in [1, 3, 5, 9]:
            for window_s in [1,2,3,5,7]:
                for rand_r in [.3333,.5,1,2,3]:
                    x = Classy(amino_acid=site, forest_size=110, classy=classy, window_size=window_s, neg_per_seq=ratio)
                    x.generate_data(random_ratio=rand_r)
                    x.calculate()
                    x.report()
                    print("rand_r: "+str(rand_r))
                    x.test(positive_file="pos.fasta", negative_file="PKA_neg.fasta")