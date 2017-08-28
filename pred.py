from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import random
from random import randint
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from imblearn.ensemble import EasyEnsemble
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.manifold import TSNE
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import os
from sklearn.metrics import classification_report


def distance(s1: str, s2: str, threshold: float =.9):
    t = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            t += 1
    if (len(s1)-t) / s2 < threshold:
        return False
    else:
        return True


def windower(sequence: str, position: int, wing_size: int):
    # final window size is wing_size*2 +1
    # Checks to make sure positions and wing_size are not floats
    position = int(position)
    wing_size = int(wing_size)
    # Logic to make sure no errors are thrown due to overhangs when slicing the sequence
    if (position - wing_size) < 0:
        return sequence[:wing_size + position]
    if (position + wing_size) > len(sequence):
        return sequence[position - wing_size:]
    else:
        return sequence[position - wing_size:position + wing_size]


def chemical_vector(temp_window: str):
    """
    This provides a feature vector containing the sequences chemical properties
    Currently this contains hydrophobicity (gravy), aromaticity, and isoelectric point
    Overall this vector does not preform well and can act as a control feature vector
    """
    temp_window = temp_window.strip("\"")
    temp_window = ProteinAnalysis(temp_window)
    return [temp_window.gravy(), temp_window.aromaticity(), temp_window.isoelectric_point()]

# noinspection PyDefaultArgument
def sequence_vector(temp_window: str, seq_size: int = 21, hydrophobicity=1, trash=["\"", "B", "X", "Z", "U", "X"]):
    """
    This vector takes the sequence and has each amino acid represented by an int
    0 represents nonstandard amino acids or as fluff for tails/heads of sequences
    Strip is a list which can be modified as user needs call for
    """
    for i in trash:
        temp_window = temp_window.replace(i, "")
    vec = []
    aa = {"G": 1, "A": 2, "L": 3, "M": 4, "F": 5, "W": 6, "K": 7, "Q": 8, "E": 9, "S": 10, "P": 11, "V": 12, "I": 13,
          "C": 14, "Y": 15, "H": 16, "R": 17, "N": 18, "D": 19, "T": 20, "X": 0}

    for i in temp_window:
        vec.append(aa[i])
    if len(vec) != seq_size:
        t=len(vec)
        for i in range(seq_size-t):
            vec.append(0)
    # Hydrophobicity is optional
    if hydrophobicity == 1:
        vec.append(ProteinAnalysis(temp_window).gravy())
    return vec


def hydrophobicity_vector(temp_window: str):
    """
    Just returns the hydrophobicity as a feature, another control vector
    """
    temp_window = temp_window.strip("\"")
    temp_window = ProteinAnalysis(temp_window)
    return [temp_window.gravy()]


def generate_random_seq(locked: list, wing_size: int, center: str):
    """
    Generates random sequences and checks that they aren't in locked
    Locked is a list of sequences which are known to be positives
    """
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
    """
    Cleans up data from various csvs with different organizational preferences
    Assumes column names are sequence, code, and position
    Enables the user to generate negative examples, sequences which aren't in the known positives are assumed to negative
    I chose to make the DataCleaner require extra steps to run since I am assuming people using it come from a
    non CS background and
    the extra steps are meant to enable easier debugging and understanding of the flow
    """
    def __init__(self, file: str, delimit: str =",", header_line: int=0, wing=10):
        """

        :param file: Input file
        :param delimit: What delimiter is used by the csv
        :param header_line: Used by pandas to determine the header
        :param wing: how long the seq is on either side of the modified amino acid
        For example wing size of 2 on X would be AAXAA
        """
        self.data = pd.read_csv(file, header=header_line, delimiter=delimit, quoting=3, dtype=object)
        self.protiens = {}
        self.count = 0
        self.labels = []
        self.sequences = []
        self.wing = wing

    def load_data(self, amino_acid: str, aa: str="code", seq: str="sequence", pos: str ="position"):
        """
        Loads the data into the object
        :param amino_acid: Which amino acid is the ptm found on
        :param aa: the column name for the amino acid modified in the PTM site
        :param seq: the column name for the FULL protien sequence
        :param pos: the column name for where the ptm occurs in the PTM,
        Assumed it is 1-based index and the code adjusts for that
        :return: loads data into the data cleaner object
        """
        for i in range(len(self.data[seq])):
            if self.data[aa][i] in amino_acid:
                try:
                    t = self.data[seq][i]
                    if t not in self.protiens.keys():
                        self.protiens[t] = [int(self.data[pos][i])]
                    else:
                        self.protiens[t] = self.protiens[t].append(int(self.data[pos][i]))
                except:
                    pass

    def generate_positive(self):
        """
        Populates the object with the positive PTM sequences, clips them down to the intended size
        :return: Populates the object with the positive PTM sequences
        """
        for i in self.protiens.keys():
            try:
                t = self.protiens[i]
                for j in t:
                    self.sequences.append(windower(sequence=i, position=j-1, wing_size=self.wing))
                    self.labels.append(1)
            except:
                pass

    def generate_negatives(self, amino_acid: str, ratio: int=-1, cross_check: int=-1):
        """
        Finds assumed negatives in the sequences, can control the ratio and whether
        :param amino_acid: The amino acid where the PTM occurs on, used for generating the negative
        :param ratio: if -1 just adds every presumed negative, otherwise use float/int value to determine
        :param cross_check: if -1 doesnt cross check otherwise it ensures
        that no negative sequences extracted match positive sequences
        WARNING: The larger the data the longer it will take
        :return: Adds negatives to data in the object
        """
        self.count = len(list(self.protiens.keys()))
        if ratio < 0:
            for i in self.protiens.keys():
                try:
                    for j in range(len(i)):
                        if i[j] == amino_acid and j+1 not in self.protiens[i]:
                            self.sequences.append(windower(sequence=i, position=j, wing_size=self.wing))
                            self.labels.append(0)
                except:
                    pass
        else:
            t = len(self.sequences)
            for y in range(int(t*ratio)):
                try:
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
                except:
                    pass

    def write_data(self, output: str, seq_col: str="sequence", label_col: str="label", shuffle=0):
        """
        Writes the data to an output file
        :param output: Output file name
        :param seq_col: column name of the sequence
        :param label_col: Column of the label of the classifier
        :param shuffle: If not 0 will randomly shuffle the data before writing it
        :return:
        """
        file = open(output, "w+")
        t = str(seq_col) + "," + str(label_col)+"\n"
        file.write(t)
        if shuffle != 0:
            temp = list(zip(self.sequences, self.labels))
            random.shuffle(temp)
            self.sequences, self.labels = zip(*temp)
        for i in range(len(self.sequences)):
            file.write(str(self.sequences[i]) + "," + str(self.labels[i]) + "\n")


class FastaToCSV:
    # More Benchmarks
    def __init__(self, negative: str, positive: str, output: str):
        write_head = 0
        if not os.path.isfile(output):
            write_head = 1
        output = open(output, "a+")
        if write_head == 1:
            output.write("sequence,label,code\n")
        negative = open(negative)
        for line in negative:
            if ">" not in line:
                line = line.replace("\n", "")
                s = line+","+str(0)+","+line[10]+"\n"
                output.write(s)
        negative.close()
        positive = open(positive)
        for line in positive:
            line = line.replace("\n", "")
            if ">" not in line:
                s = line+","+str(1)+","+line[10]+"\n"
                output.write(s)
        positive.close()
        output.close()



# noinspection PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit,PyAttributeOutsideInit
class Predictor:
    """
    The prototyping tool, meant to work with data outputted by datacleaner

    """
    def __init__(self,  window_size=7, training_ratio=.7, seq="sequence", pos="label"):
        self.training_ratio = training_ratio  # Float value representing % of data used for training
        self.features = []
        self.labels = []
        self.words = []
        self.window_size = window_size
        self.supervised_classifiers = {"forest": RandomForestClassifier(n_jobs=4),
                                       "mlp_adam": MLPClassifier(activation="logistic"),
                                       "svc": svm.SVC()}
        self.imbalance_functions = {"easy_ensemble": EasyEnsemble(), "SMOTEENN": SMOTEENN(),
                                    "SMOTETomek": SMOTETomek(), "ADASYN": ADASYN(),
                                    "random_under_sample": RandomUnderSampler(), "ncl": NeighbourhoodCleaningRule(),
                                    "near_miss": NearMiss()}
        self.seq = seq
        self.pos = pos
        self.random_data = 0
        self.test_results = 0
        self.vecs = {"sequence": sequence_vector, "chemistry": chemical_vector, "hydrophobicity": hydrophobicity_vector}
        self.vector = 0

    def load_data(self, file, delimit=",", header_line=0):
        """
        Reads the data for processing
        :param file: File name
        :param delimit: for pandas
        :param header_line: for pandas
        :return: Loads the data inot the object
        """
        # Modify these if working with different CSV column names
        data = pd.read_csv(file, header=header_line, delimiter=delimit, quoting=3, dtype=object)
        self.data = data.reindex(np.random.permutation(data.index))
        for i in range(len(data[self.seq])):
            if type(data[self.seq][i]) == str:
                self.features.append(data[self.seq][i])
                self.labels.append(data[self.pos][i])

    def generate_random_data(self, ratio, amino_acid):
        """
        Must be executed after load_data
        Adds randomly generated data that is automatically cross checked
        :param ratio: ratio of random to real data
        :param amino_acid: amino acid which the ptm is being tested for
        :return: Adds random data to the generated data
        """
        print("Loading Data")
        temp_len = len(self.features)
        self.random_seq = []
        self.random_data = 1
        for i in range(int(ratio*temp_len)):
            self.random_seq.append(generate_random_seq(center=amino_acid, wing_size=int(self.window_size*.5),
                                                            locked=self.data[self.seq]))
        print("Loaded Data")

    def vectorize(self, vectorizer):
        """
        Vectorizes the data using a vector function of choice
        Long term goal is convert this to string input and have an internal dict
        :param vectorizer: function which will vectorize the data
        :return: vectorized data
        """
        print("Applying Vector Function")
        t = []

        self.vector = self.vecs[vectorizer]
        for i in self.features:
            t.append(self.vector(i))
        self.features = t
        print("Finished Applying Vector Function")

    def balance_data(self, imbalance_function):
        """
        Applies imblearn function to the data
        :param imbalance_function: imblearn function of choice, it is a string
        :return: balanced data
        """
        print("Balancing Data")
        imba = self.imbalance_functions[imbalance_function]
        self.features, self.labels = imba.fit_sample(self.features, self.labels)
        print("Balanced Data")

    def supervised_training(self, classy: str, scale: str =-1):
        """
        Trains and tests the classifier on the data
        :param classy: Classifier of choice, is string passed through dict
        :param scale: Applies a scaler function from sklearn if not -1
        :return: Classifier trained and ready to go and some results
        """
        print("Starting Training")
        self.features = list(self.features)
        print(len(self.features), len(self.labels))
        self.classifier = self.supervised_classifiers[classy]
        temp = list(zip(self.features, self.labels))
        random.shuffle(temp)
        self.features, self.labels = zip(*temp)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.labels,
                                                                        test_size=0.3, random_state=42)
        if self.random_data > 0:
            for i in range(len(self.random_seq)):
                np.append(self.X_train, self.vector(self.random_seq[i]))
                np.append(self.y_train, 0)
        if scale != -1:
            st = {"standard":StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler(), "max": MaxAbsScaler()}
            self.X_train = st[scale].fit_transform(X=self.X_train)
            self.X_test = st[scale].fit_transform(X=self.X_test)

        self.classifier.fit(self.X_train, self.y_train)
        print("Done training")
        self.test_results = self.classifier.predict(self.X_test)
        print("Test Results")
        print(classification_report(y_pred=self.test_results, y_true=self.y_test, target_names=["Non PTM", "PTM"]))

    def benchmark(self, benchmark: str, aa: str):
        benchmark = open(benchmark)
        validation = []
        answer_key = []
        for i in benchmark:

            s = i.split(",")
            label = s[1].replace("\n", "").replace("\t", "")
            seq = s[0].replace("\n", "").replace("\t", "")
            code = s[2].replace("\n", "").replace("\t", "")

            if aa == code:
                validation.append(self.vector(seq))
                answer_key.append(int(label))
        v = self.classifier.predict(validation)
        v.reshape(len(v), 1)
        answer_key = np.asarray(answer_key)
        answer_key.reshape(len(answer_key), 1)
        t= []
        for i in v:
            t.append(int(i))
        v = np.asarray(t).reshape(len(t), 1)
        for i in range(len(answer_key)):
            if answer_key[i] != 0 and answer_key[i] != 1:
                print(i, "answer")
            if v[i] != 0 and v[i] != 1:
                print(i, "V", v[i], type(v[i]))
        print(precision_recall_fscore_support(v,  answer_key, labels=[1]))
        print(classification_report(y_pred=v, y_true=answer_key, target_names=["Non PTM", "PTM"]))

    def generate_pca(self):
        """

        :return: PCA of data
        """
        y = np.arange(len(self.features))
        pca = PCA(n_components=2)
        x_np = np.asarray(self.features)
        pca.fit(x_np)
        X_reduced = pca.transform(x_np)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='RdBu', s=1)
        plt.xlabel('First component')
        plt.ylabel('Second component')
        plt.show()

    def generate_tsne(self):
        y = np.arange(len(self.features))
        tsne = TSNE(n_components=2)
        x_np = np.asarray(self.features)
        X_reduced = tsne.fit_transform(x_np)
        plt.figure(figsize=(10, 8))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='RdBu', s=1)
        plt.xlabel('First component')
        plt.ylabel('Second component')
        plt.show()

    def test_seq(self, s: str):
        s= self.vector(s)
        return self.classifier.predict(s)

    def test_sequences(self, s: list):
        t = []
        for i in s:
            t.append(self.vector(i))
        return self.classifier.predict(t)

"""
x = DataCleaner("/Users/mark/PycharmProjects/phosphosites.csv")
x.load_data(amino_acid="H")
x.generate_positive()
x.generate_negatives(cross_check=1, amino_acid="H")
x.write_data("Data/Training/clean_h.csv")
"""
x = FastaToCSV(negative="Data/Benchmarks/Raw/PKC_neg.fasta", positive="Data/Benchmarks/Raw/PKC_pos.fasta", output="Data/Benchmarks/phos.csv")