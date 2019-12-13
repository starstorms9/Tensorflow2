#%% Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import io
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# The Embedding layer takes at least two arguments:
# the number of possible words in the vocabulary, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 32.
# embedding_layer = layers.Embedding(1000, 32)

#%% Download and prep data
vocab_size = 10000
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0])
maxlen = 500

#%% Add padding
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=maxlen)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=maxlen)

print(train_data[0])

#%% Testing
words = [ [word, num] for word, num in imdb.get_word_index().items() ]

#%% Create simple model
embedding_dim=1

model = keras.Sequential()
model.add( layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add( layers.GlobalAveragePooling1D() )
# model.add( layers.Dense(16, activation='relu') )
model.add( layers.Dense(1, activation='sigmoid') )

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%% Train model
history_new = model.fit(
    train_data,
    train_labels,
    epochs=100,
    batch_size=512,
    validation_data=(test_data, test_labels))

#%% My functions
def dist(arr, i1, i2) :
    return np.linalg.norm( arr[i1] - arr[i2] )

def dists(arr, index) :
    return [dist(arr, index, i) for i in range(arr.shape[0])]

def closest_int(arr, index : int) :
    return [ reverse_word_index[i] for i in np.argsort(dists(embs, index))[:10]]

def closest_word(arr, index : str) :
    return closest_int(arr, word_index[index])

def playClosest(embeddings) :
    while True :
        word = input("New word: ")
        print(closest_int(embeddings, word_index[word]))

def w2vec(embeddings, word) :
    return embeddings[word_index[word]][0]

def encode(text, maxlen=500) :
    text_nums = [ word_index.get(i, 2) for i in text.split(' ') ]
    return keras.preprocessing.sequence.pad_sequences( [text_nums], value=word_index["<PAD>"], padding='post', maxlen=maxlen)

#%% Playing
embs = model.layers[0].get_weights()[0]

predictions = np.round(model.predict(test_data).reshape((-1,))).astype(test_labels.dtype)
# 0 if correct, 1 if predicted good but actually bad, -1 if predicted bad but actually good
diff = predictions - test_labels
too_good = np.argwhere(diff==1)
too_bad = np.argwhere(diff==-1)

#%% Post process
history_dict = history_new.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# plt.figure(figsize=(12,9))
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim((0.5,1))
plt.show()

#%% Get embeddings
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# Can now upload those files to http://projector.tensorflow.org/ to visualize