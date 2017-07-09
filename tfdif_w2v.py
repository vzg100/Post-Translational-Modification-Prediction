import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import numpy as np
from tabulate import tabulate
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
class Classy:
    def __init__(self, data="clean_t.csv", delimit=",", amino_acid="T", training_ratio=.7, header_line=0):

        self.data = open(data, "r")
        self.amino_acid = amino_acid
        self.training_ratio = training_ratio  # Float value representing % of data used for training
        self.features= []
        self.labels = []
        i = 0
        for line in self.data:
            try:
                x, y = line.split(",")
                y = int(y.strip("\n"))
                t = []
                for j in x:
                    t.append(j)
                self.features.append(t)
                self.labels.append(y)
            except:
                print("Bad data at line"+str(i))
            i = i + 1
        temp = list(zip(self.features, self.labels))
        random.shuffle(temp)
        self.features, self.labels = zip(*temp)

        self.num_features = 300  # Word vector dimensionality
        self.min_word_count = 1  # Minimum word count
        self.num_workers = 4  # Number of threads to run in parallel
        self.context = 5  # Context window size
        self.downsampling = 5e-1  # Downsample setting for frequent words
        self.model = word2vec.Word2Vec(self.features ,workers=self.num_workers, size=self.num_features, min_count=self.min_word_count,window=self.context, sample=self.downsampling)
        w2v = {w: vec for w, vec in zip(self.model.wv.index2word, self.model.wv.syn0)}
        mult_nb = Pipeline(
            [("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
        bern_nb = Pipeline(
            [("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
        mult_nb_tfidf = Pipeline(
            [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
        bern_nb_tfidf = Pipeline(
            [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
        svc = Pipeline(
            [("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
        svc_tfidf = Pipeline(
            [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

        etree = Pipeline([("w2v vect", MeanEmbeddingVectorizer(w2v)),
                                      ("extra trees", ExtraTreesClassifier(n_estimators=200))])
        etree_tfidf = Pipeline([("tfid vectr", TfidfEmbeddingVectorizer(w2v)),
                                            ("extra trees", ExtraTreesClassifier(n_estimators=200))])
        random_forest = Pipeline([("mean vectorizer", MeanEmbeddingVectorizer(w2v)),
                              ("random forest", RandomForestClassifier(n_estimators=200))])
        random_forest_tfidf = Pipeline([("tfid vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                    ("extra trees", RandomForestClassifier(n_estimators=200))])

        all_models = [
            ("mult_nb", mult_nb),
            ("mult_nb_tfidf", mult_nb_tfidf),
            ("bern_nb", bern_nb),
            ("bern_nb_tfidf", bern_nb_tfidf),
            ("svc", svc),
            ("svc_tfidf", svc_tfidf),
            ("etree", etree),
            ("etree_tfidf", etree_tfidf),
            ("rf", random_forest),
            ("rf_tfidf", random_forest_tfidf),
        ]

        scores = sorted([(name, cross_val_score(model, self.features, self.labels, cv=5).mean())
                         for name, model in all_models],
                        key=lambda x: -x[1])
        print(tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
        plt.figure(figsize=(15, 6))
        sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])
        train_sizes = [10, 40, 160, 640, 3200, 6400]
        table = []
        for name, model in all_models:
            for n in train_sizes:
                table.append({'model': name,
                              'accuracy': benchmark(model, self.features, self.labels, n),
                              'train_size': n})
        df = pd.DataFrame(table)
        plt.figure(figsize=(15, 6))
        fig = sns.pointplot(x='train_size', y='accuracy', hue='model',
                            data=df[df.model.map(lambda x: x in ["mult_nb", "svc_tfidf", "bern_nb",
                                                                 "rf", "etree",
                                                                 ])])
        sns.set_context("notebook", font_scale=1.5)
        fig.set(ylabel="Accuracy")
        fig.set(xlabel="Labeled Training Examples")
        fig.set(title="Benchmark")
        fig.set(ylabel="Accuracy")
        plt.show()

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
                            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                                    or [np.zeros(self.dim)], axis=0)
                            for words in X
                            ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([ np.mean([self.word2vec[w] * self.word2weight[w]
                                     for w in words if w in self.word2vec] or
                                    [np.zeros(self.dim)], axis=0)
                            for words in X
                            ])
def benchmark(model, X, y, n):
    test_size = 1 - (n / float(len(y)))

    scores = []
    X = X[:n]
    y = y[:n]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

    scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
    return np.mean(scores)
x = Classy()