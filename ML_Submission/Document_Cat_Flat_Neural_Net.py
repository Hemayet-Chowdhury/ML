import glob
import errno
import codecs

import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
path1 = '/home/b-light/hc/4th yr 4th sem/ml_lab/Document_Categorization/category/accident/*.txt'
path2 = '/home/b-light/hc/4th yr 4th sem/ml_lab/Document_Categorization/category/crime/*.txt'
path3 = '/home/kingbayeed/Downloads/ML lab all dataset/Document_Categorization/category/education/*.txt'

topics, texts = [], []

files = glob.glob(path1)
# texts.append("post")
# labels.append("tags")
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            # str = re.sub(' +', ' ', str)
            str = " ".join(str.split())
            topics.append("acc")
            texts.append(str)


    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

files = glob.glob(path2)
for name in files:
    try:
        with codecs.open(name, 'r', encoding='utf-8') as f:
            str = f.read()
            #  str = re.sub(' +', ' ', str)
            str = " ".join(str.split())

            topics.append("cri")
            texts.append(str)

    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

import pandas as pd
import numpy as np

data = pd.DataFrame({'texts':texts, 'topics':topics})
#print(data.head())

from io import StringIO
col = ['topics', 'texts']
data = data[col]
data = data[pd.notnull(data['texts'])]
data.columns = ['topics', 'texts']
data['category_id'] = data['topics'].factorize()[0]
category_id_data = data[['topics', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_data.values)
id_to_category = dict(category_id_data[['category_id', 'topics']].values)
print(data.tail())

data['target'] = data.title.astype('category').cat.codes

data['num_words'] = data.data.apply(lambda x : len(x.split()))

bins=[0,50,75, np.inf]
data['bins']=pd.cut(data.num_words, bins=[0,100,300,500,800, np.inf], labels=['0-100', '100-300', '300-500','500-800' ,'>800'])

word_distribution = data.groupby('bins').size().reset_index().rename(columns={0:'counts'})

#print(word_distribution.head())

num_class = len(np.unique(data.title.values))
y = data['target'].values


MAX_LENGTH = 300
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.title.values)
post_seq = tokenizer.texts_to_sequences(data.title.values)
post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size=0.30)

vocab_size = len(tokenizer.word_index) + 1

inputs = Input(shape=(MAX_LENGTH, ))
embedding_layer = Embedding(vocab_size,
                            128,
                            input_length=MAX_LENGTH)(inputs)
x = Flatten()(embedding_layer)
x = Dense(32, activation='relu')(x)

predictions = Dense(num_class, activation='softmax')(x)
model = Model(inputs=[inputs], outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.summary()
filepath="weights-simple.hdata5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25,
          shuffle=True, epochs=5, callbacks=[checkpointer])


predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)
print("YOUR CUSTOMISED SIMPLE NEURAL NET ACCURACY IS")
print(accuracy_score(y_test, predicted))