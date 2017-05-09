from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
from random import randint
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from imblearn.ensemble import EasyEnsemble
from sklearn.model_selection import cross_val_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import  ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
print("I am running")


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

def test_suite(aa ):
    pass


def featurify(temp_window, size):
    # assumes temp_window = ProteinAnalysis(seq)
    q = [temp_window.gravy(),
         temp_window.aromaticity(),
          temp_window.isoelectric_point()
         ]
    z = temp_window.amino_acids_content
    order = {}
    counter = 0
    aa = "GALMFWKQESPVICYHRNDT"


    for i in range(len(aa)):
        order[aa[i]] = i
        counter +=1

    if len(temp_window.sequence) == size:
        for i in temp_window.sequence:
            q.append(order[i])
    else:
        for i in temp_window.sequence:
            q.append(order[i])
        for i in range(size - len(temp_window.sequence)):
            q.append(-1)
    return q




def random_seq(locked, wing_size, center):
    amino_acids = "GALMFWKQESPVICYHRNDT"
    t1, t2 = "", ""
    for i in range(wing_size):
        t1 += amino_acids[randint(0, 19)]
        t2 += amino_acids[randint(0, 19)]
    final_seq = t1 + center + t2
    if final_seq not in locked:
        return final_seq
    else:
        random_seq(locked, wing_size, center)


def report(results, answers, classy,shift=0):
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


class Classy:

    def __init__(self, data="phosphosites.csv", delimit=",", amino_acid="Y", sites="code",
                 modification="phosphorylation", window_size=7, pos="position", training_ratio=.7,
                 header_line=0, seq="sequence", neg_per_seq=5, lines_to_read=10000, classy="forest", imba=[]):
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
        self.labels = []
        self.pos_features = []
        self.neg_features = []
        self.pos_seq = []
        self.imba = imba
        self.classif = {"forest": RandomForestClassifier(verbose=0, n_jobs=4),
                           "mlp_adam": MLPClassifier(solver='adam', random_state=1, activation="logistic"),
                           "svc": svm.SVC(), "l_svc": svm.LinearSVC(),
                        "p_svc":svm.SVC(kernel="poly"),"r_svc":svm.SVC(kernel="rbf"),
                           "mlp_sgd": MLPClassifier(solver='sgd', random_state=1),
                           "mlp_lbfgs": MLPClassifier(solver='lbfgs', random_state=1),
                            "bag":BaggingClassifier(),
                            "ada":AdaBoostClassifier(svm.SVC(), algorithm="SAMME", n_estimators=200),
                            "sgd":GradientBoostingClassifier(),
                            "knn":KNeighborsClassifier(n_neighbors=1),
                            "passive_aggro": PassiveAggressiveClassifier(),"extra": ExtraTreesClassifier(),
                   "desc_tree": DecisionTreeClassifier(),"nb":GaussianNB(),"bnb":BernoulliNB(),
                        "nu_svc":svm.NuSVC(), "svr":svm.SVR(),"one_svm":svm.OneClassSVM(), "gb":GradientBoostingClassifier()}
        counter = 0
        for i in range(len(data)):
            try:
                if ("X" not in data[seq][i]) and (data[sites][i] == amino_acid) and (data[seq][i] not in self.proteins.keys()):
                    self.proteins[data[seq][i]] = [data[pos][i]]
                elif ("X" not in data[seq][i]) and (data[sites][i] == amino_acid) and (data[pos][i] not in self.proteins[data[seq][i]]):
                    self.proteins[data[seq][i]].append(data[pos][i])

                counter += 1
            except:
                pass

        for i in self.proteins.keys():
            neg_sites = []
            for position in self.proteins[i]:
                try:
                    temp_window = ProteinAnalysis(windower(i, position, self.window))
                    self.pos_seq.append(windower(i, position, self.window))
                    self.pos_features.append(featurify(temp_window, (2*self.window+1)))
                except:
                    pass
            for amino_acid_sites in range(len(i)):
                # creates list of potential negative sites from the current sequence
                if i[amino_acid_sites] == self.amino_acid and amino_acid_sites not in self.proteins[i]:
                    neg_sites.append(amino_acid_sites)
            counter = 0
            neg_sites_used = []
            while (counter < self.neg_per_seq) and (len(neg_sites_used) != len(neg_sites)):
                temp_negative = randint(0, len(neg_sites))
                if temp_negative not in neg_sites_used:
                    try:
                        temp_window = ProteinAnalysis(windower(i, temp_negative, self.window))
                        counter += 1
                        self.neg_features.append(featurify(temp_window, (2*self.window+1)))
                        self.neg_count +=1
                    except:
                        pass

    def generate_data(self, random_=1, random_ratio=2, random_test=0):
        imb_fun = {"smote":SMOTEENN(), "under":RandomUnderSampler(), "adasyn":ADASYN(), "ee":EasyEnsemble(), "smotetomek":SMOTETomek()}
        rand_features = []
        neg_labels = [0 for i in range(len(self.neg_features))]
        pos_labels = [1 for i in range(len(self.pos_features))]
        features = self.pos_features + self.neg_features
        labels = pos_labels+neg_labels
        if self.imba != []:
            for i in self.imba:
                features, labels = imb_fun[i].fit_sample(features, labels)
        if random_ == 1 and random_ratio > 0:
            for i in range(int((len(self.pos_features)+len(self.neg_features))*random_ratio)):
                rand_features.append(featurify(ProteinAnalysis(random_seq(locked=self.pos_seq, wing_size=self.window, center=self.amino_acid)), (2*self.window+1)))
        if random_test == 0:

            temp = list(zip(features, labels))
            random.shuffle(temp)
            features, labels = zip(*temp)
            training_slice = int(self.training_ratio * len(labels))
            self.training_features = list(features[:training_slice])+rand_features
            self.training_labels = list(labels[:training_slice])+[0 for i in range(len(rand_features))]

            self.test_features = features[training_slice:]
            self.test_labels = labels[training_slice:]

        else:
            features = features+rand_features

            labels = labels+[0 for i in range(len(rand_features))]
            temp = list(zip(features, labels))
            random.shuffle(temp)
            features, labels = zip(*temp)
            training_slice = int(self.training_ratio * len(labels))
            self.training_features = list(features[:training_slice])
            self.training_labels = list(labels[:training_slice])
            self.test_features = features[training_slice:]
            self.test_labels = labels[training_slice:]

    def calculate(self):
        #do a if statement type check
        t_class = []
        if type(self.classy) != list:
            self.clf = self.classif[self.classy]
        else:
            for i in self.classy:
                t_class.append((i, self.classif[i]))

            self.clf = VotingClassifier(estimators=t_class)

        self.clf.fit(self.training_features, self.training_labels)
        self.results = self.clf.predict(self.test_features)
        #self.rating = precision_recall_fscore_support(self.test_labels, self.results,average="macro")
        #print("cross val" + str(cross_val_score(self.clf, self.test_features, self.test_labels, cv=5)))
        report(answers=self.test_labels, results=self.results, classy=self.clf)

    def test(self, positive_file, negative_file, sequence_position=10):
        # for my test files sequence position = 10
        test_features = []
        test_labels = []
        with open(positive_file) as f:
            for i in f:
                if ">" not in i and i[sequence_position] == self.amino_acid:
                    temp_window = ProteinAnalysis(windower(i, sequence_position, self.window).strip("\t"))
                    feat = featurify(temp_window, (2*self.window+1))
                    test_features.append(feat)
                    test_labels.append(1)
        with open(negative_file) as f:
            for i in f:

                if ">" not in i and i[sequence_position] == self.amino_acid and "X" not in i and "U" not in i:
                    temp_window = ProteinAnalysis(windower(i, sequence_position, self.window).strip("\t"))
                    feat = featurify(temp_window, (2*self.window+1))
                    test_features.append(feat)
                    test_labels.append(0)
        temp = list(zip(test_features, test_labels))
        random.shuffle(temp)
        test_features, test_labels = zip(*temp)

        test_results = self.clf.predict(test_features)
        #print("cross val"+str(cross_val_score(self.clf, test_features, test_labels, cv=5)))
        report(results=test_results, answers=test_labels, classy=self.clf)

    def predict_seq(self, seq):
        possible_positions = []
        seq = seq.upper()
        for i in "BJOUZ":
            if i in seq:
                print("Non valid char in sequence" +i)
                return -1
        for i in range(len(seq)):
            if seq[i] == self.amino_acid:
                temp = featurify(ProteinAnalysis(windower(seq, i, self.window)), (2*self.window+1))
                possible_positions.append([i, self.clf.predict(temp)[0]])
        print(str(list(possible_positions)))

    def vis(self):
        pca = PCA(n_components=2)
        lda = LinearDiscriminantAnalysis(n_components=2)
        x_np = np.asarray(self.training_features)
        y_np = np.asarray(self.training_labels)
        #x_lda = lda.fit(x_np, y_np).transform(y_np)
        x_pca = pca.fit_transform(x_np)
        plt.figure()
        colors = ['navy',"darkorange"]

        lw = 2

        for color, i in zip(colors, [0,1]):
            plt.scatter(x_pca[y_np == i, 0], x_pca[y_np == i, 1], color=color, alpha=.5, lw=lw)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of dataset')
        plt.show()


x = Classy(data="phosphosites.csv", amino_acid="S", classy="mlp_adam", window_size=7, neg_per_seq=3,
           training_ratio=.9)
x.generate_data(random_ratio=1, random_=0)
print("Benchmark")

x.calculate()
x.test("phos_pos.fasta", "phos_neg.fasta")
