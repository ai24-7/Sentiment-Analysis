
# Deep Learning Model

#Import libraries 
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as k
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle

#Load the dataset 
df = pd.read_csv('reviews_dataset.tsv.zip',header=0, delimiter="\t", quoting=3)

# Keeping only the neccessary columns
df = df[['review','sentiment']]

#EDA 
df.shape
df.sentiment.value_counts()


# Data Preprocessing
df['review'] = df['review'].apply(lambda x: x.lower())
df['review'] = df['review'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

max_features = 1000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['review'].values)
X = tokenizer.texts_to_sequences(df['review'].values)
X = pad_sequences(X)
X.shape

with open('tokenizer.pickle', 'wb') as tk:
    pickle.dump(tokenizer, tk, protocol=pickle.HIGHEST_PROTOCOL)

y = pd.get_dummies(df['sentiment']).values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 99)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

#Model Training
embed_dim = 50
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
model.add(LSTM(10))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs = 5, verbose = 1)


# serialize model to JSON for Flask App
model_json = model.to_json()
with open("model.json", "w") as js:
    js.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")


# Test model
test = ['Movie was good but awful']
#vectorizing the tweet by the pre-fitted tokenizer instance
test = tokenizer.texts_to_sequences(test)
#padding the tweet to have exactly the same shape as `embedding_2` input
test = pad_sequences(test, maxlen=X.shape[1], dtype='int32', value=0)
print(test.shape)
sentiment = model.predict(test)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")

# EOF
