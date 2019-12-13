#%% Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras

#%% Load data
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs', split=list(splits),
    with_info=True, as_supervised=True)

print(raw_train)
print(raw_validation)
print(raw_test)

#%% Data Preprocessing
get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(5):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
   pass

image_batch.shape

trn_bat = raw_train.map(format_example)
trn_bat = trn_bat.batch(BATCH_SIZE)
for img_b, img_l in trn_bat.take(1) : pass

#%% Functions
def plotImgs(image_batch, extra_data=None, order=None, title=''):
    plt.figure(figsize=(10,9))
    plt.subplots_adjust(hspace=0.3, wspace=0.0)
    for n in range(30):
        plt.subplot(6,5,n+1)
        ordered_n = (n if (order is None) else order[n])
        plt.imshow( (image_batch[ ordered_n ]+1)/2 )
        plt.title(str(ordered_n) + (': ' + str(extra_data[ordered_n]) if extra_data is not None else ""))
        plt.axis('off')
    plt.suptitle(title)

def dist(arr, i1, i2) :
    return np.linalg.norm( arr[i1] - arr[i2] )

def dists(arr, index) :
    return [dist(arr, index, i) for i in range(arr.shape[0])]

def plotImgsRel(arr, rel_index, images, title) :
    distances = dists(arr, rel_index)
    plotImgs(images, np.round(distances, 1), np.argsort(distances), title=title)

#%% Create model base
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False
feature_batch = base_model(image_batch)
print(feature_batch.shape)

#%% Look at impact of pooling the 5 x 5 x 1280 vector
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

global_max_layer = tf.keras.layers.GlobalMaxPooling2D()
feature_batch_max = global_max_layer(feature_batch)
print(feature_batch_max.shape)

g_flatten = tf.keras.layers.Flatten()
flat_bat = g_flatten(feature_batch)
print(flat_bat.shape)

plotImgsRel(feature_batch_average.numpy(), 23, image_batch, 'Average')
plotImgsRel(feature_batch_max, 23, image_batch, 'Max')
plotImgsRel(flat_bat, 23, image_batch, 'Flat')

#%% Add classification head
prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential()
model.add(base_model)
model.add(global_average_layer)
model.add(prediction_layer)

#%% Compile model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
print(len(model.trainable_variables))
model.summary()

#%% Train the model
num_train, num_val, num_test = (
  metadata.splits['train'].num_examples*weight/10
  for weight in SPLIT_WEIGHTS
)

initial_epochs = 10
steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = 20

loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)

#%% Training evaluation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#%% Fine tuning
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

fine_tune_above = 100
for i in range(len(base_model.layers)):
  base_model.layers[i].trainable = False if (i<fine_tune_above) else True

model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10), metrics=['accuracy'])
model.summary()
print(len(model.trainable_variables))

#%% Train fine tuned model
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches, epochs=total_epochs, initial_epoch = initial_epochs, validation_data=validation_batches)

#%% Compare training data
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#%% Test against doing both at the same time initially
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)

model = tf.keras.Sequential()
model.add(base_model)
model.add(global_average_layer)
model.add(prediction_layer)

base_model.trainable = True
fine_tune_above = 100
for i in range(len(base_model.layers)):
  base_model.layers[i].trainable = False if (i<fine_tune_above) else True

model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10), metrics=['accuracy'])

#%% Train full model at once
full_tune_epochs = 10
history_full = model.fit(train_batches, epochs=full_tune_epochs, validation_data=validation_batches)

#%% Plot results
acc_full = history_full.history['accuracy']
val_acc_full = history_full.history['val_accuracy']

loss_full = history_full.history['loss']
val_loss_full = history_full.history['val_loss']

plt.plot(acc_full)
plt.plot(val_acc_full)
plt.plot(acc)
plt.plot(val_acc)