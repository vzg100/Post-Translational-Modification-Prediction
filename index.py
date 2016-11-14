import urllib.request
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


class SequenceArray:

    def __init__(self):
        self.Sequence = []

    def readtextfile(self, file):
        # for reading simple text files
        f = open(file, 'r')
        for line in f:
            if len(line) > 1:
                for i in line:
                    self.Sequence.append(i)
            else:
                self.Sequence.append(line)
        print("Read"+file+"into the sequence")

    def show(self):
        print(self.Sequence)
    # Write janitor method for the dtpb


class Janitor:
    IDdict = {}
    Fastadict = {}
    url = 'http://www.uniprot.org/uploadlists/'
    params = {
        # Defines format you are pulling the data from, in this case UniProtKBAC
        'from': 'ACC',
        # Define what to format the data as
        'to': 'P_REFSEQ_AC',
        # returns data format
        'format': 'tab',
        # queries where to pull from
        'query': ''
    }

    contact = 'mgorelik@tulane.edu'

    def __init__(self, filename):
        f = open(filename, 'r')
        i = 0
        for line in f:

            splitline = line.split(',')
            # Protein Acquisition ID : Modification Residue, site group ID, Modification Site
            if "ACC_ID" in splitline:
                pass
            else:
                protien_acquisition_id = splitline[1]
                modification_residue = splitline[4][0]
                site_group_id = splitline[5]
                modification_region = splitline[9].upper()
                # Label
                modification_region_location = int(splitline[4][1:-2])
                self.IDdict[str(i)] = [modification_residue, site_group_id, modification_region, protien_acquisition_id, modification_region_location]
                i += 1

    def fasta_collect(self):
        print(self.IDdict)
        for i in self.IDdict.keys():
            page = urllib.request.urlopen("http://www.uniprot.org/uniprot/"+self.IDdict[i][3]+".fasta")
            sequence = str(page.read())
            t = 0
            for j in sequence:
                if j == "\\":
                    break
                t += 1

            sequence = sequence[t:]
            temp = ''
            for i2 in sequence:
                if i2 not in "\\n'":
                    temp += i2
            sequence = temp
            self.Fastadict[i] = sequence
        print(self.Fastadict)

    def write_to_csv(self, filename):
        f = open(filename, 'w')
        f.write("modification_residue,site_group_id,modification_region,protien_acquisition_id,"
                "fasta_seq,modification_region_location\n")
        for i in self.IDdict.keys():

            f.write(str(self.IDdict[i][0])+","+str(self.IDdict[i][1])+","+str(self.IDdict[i][2])+","+str(self.IDdict[i][3])+","+self.Fastadict[i]+","+str(self.IDdict[i][-1])+"\n")
        print("Done Writing")


class Classifier:
    # Label
    modification_region_location = []
    # Features
    modification_residue = []
    modification_region = []
    fasta_seq = []
    # Dics
    IDdict = {}
    preVectFeatures = []
    preVectLabels = []
    FeatureTrain = []
    FeatureTest = []
    LabelTrain = []
    LableTest = []

    def __init__(self, filename):
        f = open(filename, 'r')
        for line in f:
            splitline = line.split(',')
            self.modification_region_location.append(splitline[5].replace('\n', ""))
            self.modification_residue.append(splitline[0])
            self.modification_region.append(splitline[2])
            self.fasta_seq.append(splitline[4])
        self.modification_region_location = self.modification_region_location[1:]
        self.modification_residue = self.modification_residue[1:]
        self.modification_region = self.modification_region[1:]
        self.fasta_seq = self.fasta_seq[1:]

        for i in range(len(self.modification_region)):
            self.IDdict[i] = [self.modification_region[i], self.modification_residue[i], self.fasta_seq[i]]
        for i in range(len(self.modification_region_location)):
            self.modification_region_location[i] = int(self.modification_region_location[i])

        for i in range(len(self.modification_region)):
            self.preVectFeatures.append({"fasta_seq": self.fasta_seq[i],
                                         "modification_residue": self.modification_residue[i],
                                         "modification_region": self.modification_region[i]})
        self.preVectLabels = self.modification_region_location

    def split_me(self):
        self.FeatureTrain = self.preVectFeatures[0:90]
        self.LabelTrain = self.preVectLabels[0:90]
        self.FeatureTest = self.preVectFeatures[90:-1]
        self.LableTest = self.preVectLabels[90: -1]

    def get_classy(self):
        vec = DictVectorizer()
        train_data_features = vec.fit_transform(self.FeatureTrain)
        train_data_features = train_data_features.toarray()
        train_data_labels = np.array(self.LabelTrain).reshape((90, 1))
        test_data_features = vec.transform(self.FeatureTest)
        test_data_features = test_data_features.toarray()
        forest = RandomForestClassifier(n_estimators=50)
        classy = svm.SVC()
        neigh = KNeighborsClassifier(n_neighbors=3)
        # Training
        classy.fit(train_data_features, train_data_labels.ravel())
        forest.fit(train_data_features, train_data_labels.ravel())
        neigh.fit(train_data_features, train_data_labels.ravel())
        # Testing
        classy_predict = classy.predict(test_data_features)
        forest_predict = forest.predict(test_data_features)
        neigh_predict = neigh.predict(test_data_features)
        s = 0
        f = 0
        n = 0
        for i in range(len(self.LableTest)):

            if classy_predict[i] == self.LableTest[i]:
                s += 1
            if forest_predict[i] == self.LableTest[i]:
                f += 1
            if neigh_predict[i] == self.LableTest[i]:
                n += 1
        if s != 0:
            s = s/len(self.LableTest)
            print("SVC has a success rate of:"+ str(s))
        else:
            print("SVC Failed all Test Cases")
        if f != 0:
            f = f/len(self.LableTest)
            print("Random Forest has a success rate of:" + str(f))
        else:
            print("Random Forest Failed all Test Cases")
        if n != 0:
            n = n/len(self.LableTest)
            print("KNN has a success rate of:" + str(n))

        else:
            print("KNN Failed all Test Cases")
x = Classifier("temp_test.csv")
x.split_me()
x.get_classy()
# Short Term
# TODO: Clean up code
# TODO: Scrape additional data: Lociliazation, AA specificity, motiff specificity
# TODO: Modify parameters