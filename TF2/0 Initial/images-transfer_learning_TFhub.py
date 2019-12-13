#%% Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
from PIL import Image
import time

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from easy_tf_log import tflog

import numpy as np

#%% Imagenet classifier
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([ hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,)) ])

#%% Run on single image
grace_hopper_path = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper_path).resize(IMAGE_SHAPE)

rotations = [90, 180, 270, 20, 40, 60, 120, -60]
grace_hopper_imgs = [grace_hopper.rotate(rot) for rot in rotations]
grace_hoppers = [np.array(gc_img)/255.0 for gc_img in grace_hopper_imgs]
gc_batch = np.stack(grace_hoppers)

result = classifier.predict(gc_batch)
result.shape

predicted_classes = [np.argmax(result[i], axis=-1) for i in range(len(grace_hoppers))]
predicted_classes

#%% Decode prediction
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

index = 2
plt.imshow(grace_hoppers[index])
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_classes[index]]
_ = plt.title("Prediction: " + predicted_class_name.title())

predictions = [imagenet_labels[predicted_classes[i]] for i in range(len(predicted_classes))]
print(predictions)

#%% Download flower data
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

#%% Transfer Learning Setup
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

#%% Make initial predictions
result_batch = classifier.predict(image_batch)
result_batch.shape

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
predicted_class_names

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title( str(n) + ":" + str(predicted_class_names[n]) )
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")

#%% Headless model
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224,224,3))

feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)
feature_extractor_layer.trainable = False

model = tf.keras.Sequential()
model.add(feature_extractor_layer)
model.add(layers.Dense(image_data.num_classes, activation='softmax'))
model.summary()

def dist(arr, i1, i2) :
    return np.linalg.norm( arr[i1] - arr[i2] )

#%% Callbacks
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

#%% Transfer learn
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
batch_stats_callback = CollectBatchStats()
history = model.fit(image_data, epochs=2, steps_per_epoch=steps_per_epoch, callbacks = [batch_stats_callback])

#%% Plot results
plt.xlabel("Training Steps")
plt.ylim([0, 1.5])
plt.grid(True)

loss_plt, = plt.plot(batch_stats_callback.batch_losses, label='Loss')
acc_plot, = plt.plot(batch_stats_callback.batch_acc, label='Acc')
plt.legend(handles=[acc_plot, loss_plt], loc=0)

#%% Check predictions
class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
class_names

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.3)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")

#%% Export model
t = time.time()

export_path = "/tmp/saved_models/{}".format(int(t))
tf.keras.experimental.export_saved_model(model, export_path)

export_path

reloaded = tf.keras.experimental.load_from_saved_model(export_path, custom_objects={'KerasLayer':hub.KerasLayer})

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)