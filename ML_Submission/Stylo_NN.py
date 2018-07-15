import keras
import matplotlib
import numpy as np
from jedi.refactoring import inline
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
# plt.style.use('ggplot')
# #%matplotlib inline
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

data = pd.read_csv('/home/b-light/hc/thesis/stackoverflowdata/bangla.csv');

#print(data.head())
print(data.tags.value_counts())
data['target'] = data.tags.astype('category').cat.codes
#print(data['target'] )
data['num_words'] = data.post.apply(lambda x : len(x.split()))
#
#print(data['num_words'] )
#
bins=[0,50,75, np.inf]
data['bins']=pd.cut(data.num_words, bins=[0,100,300,500,800, np.inf], labels=['0-100', '100-300', '300-500','500-800' ,'>800'])
#
word_distribution = data.groupby('bins').size().reset_index().rename(columns={0:'counts'})
#
print(word_distribution.head())
#
sns.barplot(x='bins', y='counts', data=word_distribution).set_title("Word distribution per bin")
print(data.head())

num_class = len(np.unique(data.tags.values))
y = data['target'].values
#
MAX_LENGTH = 2000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.post.values)
post_seq = tokenizer.texts_to_sequences(data.post.values)
post_seq_padded = pad_sequences(post_seq, maxlen=MAX_LENGTH)
#
#
X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size=0.05)
#
vocab_size = len(tokenizer.word_index) + 1




##word embeddings


#neural net begins here

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
filepath="weights-simple.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit([X_train], batch_size=64, y=to_categorical(y_train), verbose=1, validation_split=0.25,
          shuffle=True, epochs=5, callbacks=[checkpointer])
print(history)

predicted = model.predict(X_test)
predicted = np.argmax(predicted, axis=1)

print("Normal Neural Net accuracy")
print(accuracy_score(y_test, predicted))