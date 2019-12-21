#%% Imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import pandas as pd
import seaborn as sns

import pathlib
import shutil
import tempfile
from itertools import cycle

np.set_printoptions(precision=3, suppress=True)

#%%
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

#%% Get data
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')
FEATURES = 28
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")

def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

packed_ds = ds.batch(10000).map(pack_row).unbatch()

#%% Look at records
samples = list(packed_ds.take(1000))
samples = np.array([samples[i][0].numpy() for i in range(len(samples))])
plt.hist(samples.flatten(), bins=101)

#%% Look more at records
for features,label in packed_ds.batch(1000).take(1):
   plt.hist(features.numpy().flatten(), bins = 101)
   print(features.numpy().flatten().shape)
   plt.show()

#%%
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
train_ds

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

#%% Demo overfitting
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')

#%% Model functions
def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

#%% Tiny model
tiny_model = tf.keras.Sequential()
tiny_model.add(layers.Dense(16, activation='elu', input_shape=(FEATURES,)))
tiny_model.add(layers.Dense(1, activation='sigmoid'))

#%% Train tiny model
size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

#%% Check tiny model
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

#%% Define small model
small_model = tf.keras.Sequential()
# `input_shape` is only required here so that `.summary` works.
small_model.add(layers.Dense(16, activation='elu', input_shape=(FEATURES,)))
small_model.add(layers.Dense(16, activation='elu'),)
small_model.add(layers.Dense(1, activation='sigmoid'))

#%% Train small model
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

#%% Define medium model
medium_model = tf.keras.Sequential()
medium_model.add(layers.Dense(64, activation='elu', input_shape=(FEATURES,)))
medium_model.add(layers.Dense(64, activation='elu'))
medium_model.add(layers.Dense(64, activation='elu'))
medium_model.add(layers.Dense(1, activation='sigmoid'))

#%% Train medium model
size_histories['Medium']  = compile_and_fit(medium_model, "sizes/Medium")

#%% Define large model
large_model = tf.keras.Sequential()
large_model.add(layers.Dense(512, activation='elu', input_shape=(FEATURES,)))
large_model.add(layers.Dense(512, activation='elu'))
large_model.add(layers.Dense(512, activation='elu'))
large_model.add(layers.Dense(512, activation='elu'))
large_model.add(layers.Dense(1, activation='sigmoid'))

#%% Train large model
size_histories['large'] = compile_and_fit(large_model, "sizes/large")

#%% Explore model differences
plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy', smoothing_std=10)
plt.figure(figsize=(8, 8))
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
# plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")

#%% Explore model differences
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plt.figure(figsize=(8, 8))
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
# plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")

#%% View in tensorboard
# tensorboard --logdir {logdir}/sizes
display.IFrame(
    src="https://tensorboard.dev/experiment/vW7jmmF9TmKmy3rbheMQpw/#scalars&_smoothingWeight=0.97",
    width="100%", height="800px")

#%% Preventing overfitting
shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']

#%% Define L2 model
l2_model = tf.keras.Sequential()
l2_model.add(layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001), input_shape=(FEATURES,)))
l2_model.add(layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)))
l2_model.add(layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)))
l2_model.add(layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)))
l2_model.add(layers.Dense(1, activation='sigmoid'))

#%% Train L2 model
regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")

#%% Plot L2 model data (accuracy)
plotter = tfdocs.plots.HistoryPlotter(metric = 'accuracy', smoothing_std=10)
plotter.plot(regularizer_histories)
plt.ylim([0.6, 0.73])

result = l2_model(features)
regularization_loss = tf.add_n(l2_model.losses)

# Display maximum weights for large model compared to L2 model
for i in range(len(lmw)) : print("{} {:.3f} {:.3f}".format(i, np.max(lmw[i]), np.max(l2w[i])))

#%% Plot L2 model data (loss)
plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(regularizer_histories)
plt.ylim([0.55, 0.7])

result = l2_model(features)
regularization_loss = tf.add_n(l2_model.losses)

#%% Define dropout model
dropout_model = tf.keras.Sequential()
dropout_model.add(layers.Dense(512, activation='elu', input_shape=(FEATURES,)))
dropout_model.add(layers.Dropout(0.5))
dropout_model.add(layers.Dense(512, activation='elu'))
dropout_model.add(layers.Dropout(0.5))
dropout_model.add(layers.Dense(512, activation='elu'))
dropout_model.add(layers.Dropout(0.5))
dropout_model.add(layers.Dense(512, activation='elu'))
dropout_model.add(layers.Dropout(0.5))
dropout_model.add(layers.Dense(1, activation='sigmoid'))

#%% Train dropout model
regularizer_histories['dropout'] = compile_and_fit(dropout_model, "regularizers/dropout")

#%% DP model results
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

#%% Define L2 + DP model
combined_model = tf.keras.Sequential()
combined_model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu', input_shape=(FEATURES,)))
combined_model.add(layers.Dropout(0.5))
combined_model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'))
combined_model.add(layers.Dropout(0.5))
combined_model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'))
combined_model.add(layers.Dropout(0.5))
combined_model.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'))
combined_model.add(layers.Dropout(0.5))
combined_model.add(layers.Dense(1, activation='sigmoid'))

#%% Train combo model
regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")

#%% Plot histories
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

#%% Tensorboard
# %tensorboard --logdir {logdir}/regularizers
display.IFrame(
    src="https://tensorboard.dev/experiment/fGInKDo8TXes1z7HQku9mw/#scalars&_smoothingWeight=0.97",
    width = "100%",
    height="800px")