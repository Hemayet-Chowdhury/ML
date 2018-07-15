import glob
import errno
import codecs
import re

import sklearn
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, numpy, textblob, string
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

path1 = 'D:/Study/Data/Document_Categorization_Main/accident/*.txt'
path2 = 'D:/Study/Data/Document_Categorization_Main/art/*.txt'
path3 = 'D:/Study/Data/Document_Categorization_Main/crime/*.txt'
path4 = 'D:/Study/Data/Document_Categorization_Main/economics/*.txt'
path5 = 'D:/Study/Data/Document_Categorization_Main/education/*.txt'
path6 = 'D:/Study/Data/Document_Categorization_Main/entertainment/*.txt'
path7 = 'D:/Study/Data/Document_Categorization_Main/environment/*.txt'
path8 = 'D:/Study/Data/Document_Categorization_Main/international/*.txt'
path9 = 'D:/Study/Data/Document_Categorization_Main/opinion/*.txt'
path10= 'D:/Study/Data/Document_Categorization_Main/politics/*.txt'
path11= 'D:/Study/Data/Document_Categorization_Main/science_tech/*.txt'
path12= 'D:/Study/Data/Document_Categorization_Main/sports/*.txt'

labels, texts = [], []
labels1, texts1 = [], []

files = glob.glob(path1)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(1)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path2)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(2)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path3)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(3)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path4)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(4)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path5)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(5)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path6)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(6)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path7)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(7)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path8)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(8)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path9)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(9)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path10)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(10)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path11)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(11)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path12)
for name in files:
    try:
        with codecs.open(name,'r',encoding='utf-8') as f:
            str = f.read()
          #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            labels.append(12)
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise


trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.3, random_state=42)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    #print(predictions)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return metrics.accuracy_score(predictions, valid_y)

#Naive Bayes
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print ("NB, Count Vectors: ", accuracy)

#SVM
accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print ("SVM, Count Vectors: ", accuracy)

#KNN
accuracy = train_model(sklearn.neighbors.KNeighborsClassifier(n_neighbors=1), xtrain_count, train_y, xvalid_count)
print ("KNN, Count Vectors: ", accuracy)

#Random Forest
accuracy = train_model(RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print ("random forest, Count Vectors: ", accuracy)

#Descision Tree
accuracy = train_model(sklearn.tree.DecisionTreeClassifier(), xtrain_count, train_y, xvalid_count)
print ("Descision Tree, Count Vectors: ", accuracy)