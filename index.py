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
     temp_window.secondary_structure_fraction()[0],temp_window.secondary_structure_fraction()[1],temp_window.secondary_structure_fraction()[2]]
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

        tpr = tp / (tp + fn) #aka recall aka true positive rate
        spc =tn / (tn+fp) #specificty or true negative rate
        ppv = tp / (tp + fp)  # positive predicative value aka precision
        npv = tn/(tn+fn) #negative predictive value
        fpr = fp/(fp+tn) # false positive rate aka fallout
        fnr = fn/(tp+fn) #false negative rate
        fdr = fp/(tp+fp)# false discovey rate
        acc = (tp + tn) / (tp + fp + tn + fn)
        #roc = roc_auc_score(answers, results[shift:])
        inf = (tpr+spc)-1
        mkd = (ppv+npv)-1
        print("Recall:"+str(tpr))
        print("Specificity :" + str(spc))
        print("Precision:" + str(ppv))
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
        self.protiens = {}  # sequence: positive sites
        self.features = []
        self.labels = []
        self.pos_features = []
        self.pos_labels = []
        self.neg_features = []
        self.neg_labels = []
        self.pos_seq = []
        self.window = window_size
        self.neg_per_seq = neg_per_seq
        self.neg_count = 0
        # Cleans data of any sequences with unknown amino acids
        for i in range(lines_to_read):
            if ("X" not in data[seq][i]) and (data[sites][i] == amino_acid) and (data[seq][i] not in self.protiens.keys()):
                self.protiens[data[seq][i]] = [data[pos][i]]
            elif ("X" not in data[seq][i]) and (data[sites][i] == amino_acid) and (
                data[pos][i] not in self.protiens[data[seq][i]]):
                self.protiens[data[seq][i]].append(data[pos][i])

        for i in self.protiens.keys():
            neg_sites = []
            for position in self.protiens[i]:
                # Calculated positive features
                temp_window = ProteinAnalysis(windower(i, position, self.window))
                self.pos_seq.append(windower(i, position, self.window))
                feat = featurify(temp_window)
                self.features.append(feat)
                self.labels.append(1)
                self.pos_seq.append(windower(i, position, self.window))
            for amino_acid_sites in range(len(i)):
                # creates list of potential negative sites from the current sequence
                if i[amino_acid_sites] == self.amino_acid and amino_acid_sites not in self.protiens[i]:
                    neg_sites.append(amino_acid_sites)
            counter = 0
            neg_sites_used = []
            while (counter < self.neg_per_seq) and (len(neg_sites_used) != len(neg_sites)):
                temp_negative = randint(0, len(neg_sites))
                if temp_negative not in neg_sites_used:
                    temp_window = ProteinAnalysis(windower(i, temp_negative, self.window))
                    counter += 1
                    feat = featurify(temp_window)
                    self.features.append(feat)
                    self.labels.append(0)
                    self.neg_count +=1
        temp = list(zip(self.features, self.labels))
        random.shuffle(temp)
        self.features, self.labels = zip(*temp)

    def random_training(self):
        random_negative_seq = []

        for i in range(self.neg_count):
            random_negative_seq.append(random_seq(self.pos_seq, self.window, self.amino_acid))

        self.features = []
        self.labels = []
        for i in random_negative_seq:
            self.labels.append(0)
            temp_window = ProteinAnalysis(i)
            feat = featurify(temp_window)
            self.features.append(feat)
        for i in self.pos_seq:
            self.labels.append(1)
            temp_window = ProteinAnalysis(i)
            feat = featurify(temp_window)
            self.features.append(feat)

        temp = list(zip(self.features, self.labels))
        random.shuffle(temp)
        self.features, self.labels = zip(*temp)

    def mixed_training(self):
        random_negative_seq = []
        self.labels = list(self.labels)
        self.features = list(self.features)
        for i in range(self.neg_count):
            random_negative_seq.append(random_seq(self.pos_seq, self.window, self.amino_acid))
        for i in random_negative_seq:

            self.labels.append(0)
            temp_window = ProteinAnalysis(i)
            feat = featurify(temp_window)
            self.features.append(feat)

        temp = list(zip(self.features, self.labels))
        random.shuffle(temp)
        self.features, self.labels = zip(*temp)



    def calculate(self):
        self.training_slice = int(self.training_ratio * len(self.labels))
        classif = {"forest": RandomForestClassifier(verbose=0, n_jobs=4),
               "mlp": MLPClassifier(solver='adam', random_state=1),
               "svc": svm.SVC(), "l_svc": svm.LinearSVC()}
        self.clf = classif[self.classy]
        self.clf.fit(self.features[0:self.training_slice], self.labels[0:self.training_slice])
        self.results = self.clf.predict(self.features[self.training_slice:])
        self.rating = precision_recall_fscore_support(self.labels[self.training_slice:], self.results, average="macro")

    def report(self):
        print("Report for this run\n" +
          "Amino Acid: " + self.amino_acid + "\n" +
          " Classy: " + self.classy + "\n" +
          " Pos-Neg: " + str(self.neg_per_seq) + "\n" +
          " window_size: " + str(self.window) + "\n")
        report(self.results, self.labels,shift=self.training_slice)


    def speak_to_the_trees(self):
        feat_imp = self.clf.feature_importances_
        print(feat_imp)
        return feat_imp

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





for site in "S":
    for classy in ["mlp", "forest", "svc"]:
        for ratio in [1, 3, 5, 9]:
            for window_s in [1, 3, 7]:
                print("Random Training Results")
                x = Classy(amino_acid=site, forest_size=110, classy=classy, window_size=window_s, neg_per_seq=ratio)
                x.mixed_training()
                x.calculate()
                x.report()
                x.test("PKC_pos.fasta", "PKC_neg.fasta")

                #random plus non random data + class imbalnce for training, implement consensus sequence as a feature
                #try making 2/3 data random?
                #try making 1/3 data random
                #try different mlp parameters and random forest weighing
