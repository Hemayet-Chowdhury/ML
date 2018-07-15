import pandas as pd
df = pd.read_csv('/home/b-light/hc/4th yr 4th sem/ml_lab/sentiment_analysis/Sentiment Analysis/sentiment.csv')


from io import StringIO
col = ['data', 'title']
df = df[col]
df = df[pd.notnull(df['data'])]
df.columns = ['data', 'title']


##for data analysis and shape

df['class_num'] = df['title'].factorize()[0]

class_num_df = df[['title', 'class_num']].drop_duplicates().sort_values('class_num')
category_to_id = dict(class_num_df.values)
id_to_category = dict(class_num_df[['class_num', 'title']].values)

print(sentiment_counts = df.title.value_counts())


#analysis and shape ends here

import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def preprocessing(text):

    tokens = nltk.word_tokenize(text)
   
    return tokens




df['normalized_data'] = df.data.apply(preprocessing)



from nltk import ngrams
def ngrams(input_list):
    #onegrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams
df['grams'] = df.normalized_data.apply(ngrams)


import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt


import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))

vectorized_data = count_vectorizer.fit_transform(df.data)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))


def sentiment2target(sentiment):
    return {
        'Like (ভাল)': 0,
        'Smiley (স্মাইলি)': 1,
        'HaHa(হা হা)' : 2,
        'Sad (দু: খিত)': 3,
        'Skip ( বোঝতে পারছি না )': 4,
        'Love(ভালবাসা)': 5,
        'WOW(কি দারুন)': 6,
        'Blush(গোলাপী আভা)': 7,
        'Consciousness (চেতনাবাদ)': 8,
        'Rocking (আন্দোলিত হত্তয়া)': 9,
        'Bad (খারাপ)': 10,
        'Angry (রাগান্বিত)': 11,
        'Fail (ব্যর্থ)': 12,
        'Provocative (উস্কানিমুলক)': 13,
        'Shocking (অতিশয় বেদনাদায়ক)': 14,
        'Protestant (প্রতিবাদমূলক)': 15,
        'Evil (জঘন্য)': 16,
        'Skeptical (সন্দেহপ্রবণ)': 17,
    }[sentiment]



targets = df.title.apply(sentiment2target)

from sklearn.model_selection import train_test_split
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.2, random_state=0)
data_train_index = data_train[:,0]
data_train = data_train[:,1:]
data_test_index = data_test[:,0]
data_test = data_test[:,1:]
#




from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn import tree, svm
from sklearn.naive_bayes import MultinomialNB




print("naive bayes ")
clf = MultinomialNB().fit(data_train, targets_train)
print(clf.score(data_test, targets_test))


print("Decision tree")
#decision tree starts here
clf = tree.DecisionTreeClassifier()
clf_output= clf.fit(data_train, targets_train)
print(clf.score(data_test, targets_test))

print("SVM with linear kernel ")
#svm starts here
from sklearn import svm
clf = svm.SVC()
clf.fit(data_train, targets_train)
print(clf.score(data_test, targets_test))



#knn starts here
print("KNN ")

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data_train, targets_train)

print(neigh.score(data_test, targets_test))
