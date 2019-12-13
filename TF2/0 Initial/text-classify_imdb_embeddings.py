#%% Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import io
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#%% Download data
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
len(train_data[0]), len(train_data[1])

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#%% Functions
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

def dist(arr, i1, i2) :
    return np.linalg.norm( arr[i1] - arr[i2] )

def dists(arr, index) :
    return [dist(arr, index, i) for i in range(arr.shape[0])]

def closest_int(arr, index : int) :
    return [ reverse_word_index[i] for i in np.argsort(dists(arr, index))[:10]]

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

#%% Prep data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

len(train_data[0]), len(train_data[1])
print(train_data[0])

#%% Build model
vocab_size = 10000
embedding_dim = 16

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_dim))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = dict()

#%% Validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#%% Active plot callback
history = dict()
plot_keys = ['accuracy', 'val_accuracy']

# Class for displaying progress on the end of an epoch
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global history
        for key in logs.keys() :
            if key in history.keys() : history[key].append(logs[key])
            else : history[key] = [logs[key]]

        for key in plot_keys : plt.plot(history[key], label = key)
        plt.legend(loc = 'upper left')
        plt.show()
pass

#%% Train model
history_train = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512,
                    validation_data=(x_val, y_val), verbose=1, callbacks = [DisplayCallback()])

#%% Evaluate
results = model.evaluate(test_data, test_labels)
print(results)

#%% Plot results
history_dict = history_train.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

