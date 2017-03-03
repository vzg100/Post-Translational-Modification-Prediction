import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
from random import randint
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

def windower(seq, pos, w):
    pos = int(pos)
    w = int(w)
    if (pos - w) < 0:
        return seq[:w+pos]
    if (pos + w) > len(seq):
        return seq[pos-w:]
    else:
        return seq[pos-w:pos+w]
    
class Classy:
    def __init__(self, data="phosphosites.csv", delimit=",", amino_acid="Y", sites="code",
                 modification="phosphorylation", window_size=7, pos="position", training_ratio=.7,
                 header_line=0, seq="sequence", neg_per_seq = 10, lines_to_read=90000, forest_size=110, classy="forest"):
        self.classy = classy
        data = pd.read_csv(data, header=header_line, delimiter=delimit, quoting=3, dtype=object)
        self.data = data.reindex(np.random.permutation(data.index))
        self.amino_acid = amino_acid
        self.training_ratio = training_ratio
        self.protiens = {}
        self.features = []
        self.labels = []
        self.pos_features = []
        self.pos_labels = []
        self.neg_features = []
        self.neg_labels = []
        for i in range(lines_to_read):
            if("X" not in data[seq][i]) and (data[sites][i] == amino_acid) and (data[seq][i] not in self.protiens.keys()):
                self.protiens[data[seq][i]]=[data[pos][i]]
            elif("X" not in data[seq][i]) and (data[sites][i] == amino_acid) and (data[pos][i] not in self.protiens[data[seq][i]]):
                self.protiens[data[seq][i]].append(data[pos][i])
            else:
                pass
        #neg_per_seq = len(self.protiens.keys())
        for i in self.protiens.keys():
            neg_pos = []
            neg_pos_used = []
            
            positions = self.protiens[i]
            for p in positions:
                window = ProteinAnalysis(windower(i, p, window_size))
                self.features.append([window.gravy(), window.aromaticity(), window.isoelectric_point()])
                self.pos_features.append([window.gravy(), window.aromaticity(), window.isoelectric_point()])
                self.labels.append(1)
                self.pos_labels.append(1)
            for aa in range(len(i)):
                if (i[aa] == amino_acid) and (aa not in positions):
                    neg_pos.append(aa)
            counter = 0
            while(counter < neg_per_seq) and (len(neg_pos_used) != len(neg_pos)):
                qq = randint(0, len(neg_pos))
                if(qq not in neg_pos_used):
                    window = ProteinAnalysis(windower(i, qq, window_size))
                    self.features.append([window.gravy(), window.aromaticity(), window.isoelectric_point()])
                    self.labels.append(0)   
                    self.neg_labels.append(0)
                    self.neg_features.append([window.gravy(), window.aromaticity(), window.isoelectric_point()])
                    neg_pos_used.append(qq)
                    counter += 1

        temp = list(zip(self.features, self.labels))
        random.shuffle(temp)
        self.features, self.labels = zip(*temp) 
    def calculate(self):
        self.training_slice = int(self.training_ratio*len(self.labels))
        if self.classy == "forest":
            
            self.forest = RandomForestClassifier(verbose = 0, n_jobs = 4)
            self.forest = self.forest.fit(self.features[0:self.training_slice], self.labels[0:self.training_slice])
            self.results = self.forest.predict(self.features[self.training_slice:])
            self.rating = precision_recall_fscore_support(self.labels[self.training_slice:], self.results, average="macro")
        if self.classy == "mlp":
            self.clf = MLPClassifier(solver='adam', random_state=1)
            self.clf.fit(self.features[0:self.training_slice], self.labels[0:self.training_slice])
            self.results = self.clf.predict(self.features[self.training_slice:])
            self.rating = precision_recall_fscore_support(self.labels[self.training_slice:], self.results, average="macro")
    
    def report(self):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        print("Report for Amino Acid: "+self.amino_acid)
        for i in range(len(self.results)):
            if self.results[i] == 1 and self.labels[i+self.training_slice] == 1:
                tp+=1
            elif self.results[i] == 0 and self.labels[i+self.training_slice] == 0:
                tn +=1 
            elif self.results[i] == 1 and self.labels[i+self.training_slice] == 0:
                fp +=1
            else:
                fn+=1
        if tp != 0:
            self.prec = tp/(tp+fp) # sensitivity 
            self.recall = tp/(tp+fn)
            self.acc = (tp+tn)/(tp+fp+tn+fn)
            print(self.prec, self.recall, self.acc)
            self.roc = roc_auc_score(self.labels[self.training_slice:], self.results)
            print(self.roc)
            return([self.roc, self.prec, self.recall, self.acc])
        else:
            print("Failed")
            return False
        #count positives vs negatives and where they pop up, look for feedback places online 
        #take run of the mill problem and compare to accruacy 
        #check cbs paper 
    def save_to_csv(self, filname):
        f = open(filename, "w")
        f.write(self.amino_acid+","+self.prec+","+self.recall+","+self.acc+"\n")
    def speak_to_the_trees(self):
        feat_imp = self.forest.feature_importances_
        print(feat_imp)
        return(feat_imp)
    def vote(self, voters, pos_used, neg_ratio):
        #voters is how many classifiers will be createt
        #pos_used % of positive examples used to train, must be less than .95
        #neg_ratio % of each classifiers training will be negative examples 
        
        #TODO: Randomly shuffle negative & positive trianing sets
        length_voter = int(pos_used*len(self.pos_features))/(1-neg_ratio)
        
        partition_count = 
        neg_combos = [self.neg_features[i:i+partition_count] for i in range(0, len(self.neg_features), partition_count)]
        vote_training_features = []
        vote_training_labels = []
        vote_test_features = []
        vote_test_labels = []
        for i in range(len(neg_combos)-1):
for i in "SYTK":
    x = Classy(amino_acid=i, forest_size=110, classy="mlp")
    x.calculate()
    x.report()
    y = Classy(amino_acid=i, forest_size=110, classy="forest")
    y.vote(5)
    y.calculate()
    y.report()
