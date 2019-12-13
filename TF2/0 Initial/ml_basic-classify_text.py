#%% Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds
np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt
import easy_tf_log as etl
from datetime import datetime

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

#%% Get Data
# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews", 
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

#%% Explore Data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch
train_labels_batch

#%% Embedding models to try
embed_num = 1
embeddings = ["https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
              "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1",
              "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
              "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"]

#%% Build model from hub
embed = hub.load(embeddings[embed_num])
embed_dim = embed.embeddings.get_shape()[1]

with tf.device('/cpu:0'):
    hub_layer = hub.KerasLayer(embeddings[embed_num], output_shape=(embed_dim,), input_shape=[], dtype=tf.string, trainable=True)
    # hub_layer(train_examples_batch[:3]) 
    
#%% Build Full model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#%% Tensorboard setup
logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#%% Train model
history = model.fit(train_data.shuffle(10000).batch(512), epochs=20, validation_data=validation_data.batch(512), verbose=1, callbacks=[tensorboard_callback])

#%% Eval model
results = model.evaluate(test_data.batch(512), verbose=0)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
  
#%% Plot Graphs
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
pass

#%% Testing
rev = ['This movie was bad',
       'This movie was good',
       'This movie was not bad',
       'This movie was not good',
       'I did not like this movie',
       'Why did I waste my time watching this movie',
       'I hate this movie but love the actors',
       'This movie is incredibly boring',
       'Although the acting was good I did not like this movie',
       'This has to be the best movie I have ever seen, it is amazing',
       'This has to be the best movie I have ever seen, it is not amazing']

rev_pred = model.predict(tf.convert_to_tensor(rev))

revs = [rev, rev_pred]

def print_rev(index) :
    print(revs[0][index], "\nScore", revs[1][index][0])
    
for i in range(len(rev)) :
    print_rev(i)
    
#%% Random input
for i in range(10) :
    new_rev = input()
    prediction = model.predict(tf.convert_to_tensor([new_rev]))
    print(prediction[0])