#%% Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

np.set_printoptions(precision=3, suppress=True)
tfds.disable_progress_bar()
keras = tf.keras

#%% Data pre
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs', split=list(splits),
    with_info=True, as_supervised=True)

print(raw_train)
print(raw_validation)
print(raw_test)

get_label_name = metadata.features['label'].int2str

#%% Show some images
for image, label in raw_train.shuffle(100).take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

#%% Show formatted pic method
def show_pic(image) :
    plt.xticks([])
    plt.yticks([])
    plt.imshow( (image+1) / 2 )
    
def unformat_pic(image) :
    return (image+1)/2

#%% Format pic
def format_pic(image):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image

imex = [image for image, label in raw_train.shuffle(100).take(5)]
imexf = list(map(format_pic, imex))
  
#%% Format data
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

for image_batch, label_batch in train_batches.take(1): pass

print(image_batch.shape)

#%% Create model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

feature_batch = base_model(image_batch)
print(feature_batch.shape)

#%% Feature extraction
base_model.trainable = False
base_model.summary()

#%% Add classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential()
model.add(base_model)
model.add(global_average_layer)
model.add(prediction_layer)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
len(model.trainable_variables)

#%% Train model
num_train, num_val, num_test = (
  metadata.splits['train'].num_examples*weight/10
  for weight in SPLIT_WEIGHTS
)

initial_epochs = 10
steps_per_epoch = round(num_train)//BATCH_SIZE
validation_steps = 20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

#%% Train
history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)

#%% Evaluate
loss1,accuracy1 = model.evaluate(validation_batches, steps = validation_steps)
print("initial loss: {:.2f}".format(loss1))
print("initial accuracy: {:.2f}".format(accuracy1))

#%% Learning analysis
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

#%% Explore results
oClass = model.layers[-2].output  # global mapping 2D layer
mClass = keras.Model(model.input, outputs=oClass)

cw = model.get_layer('dense').kernel.numpy()[:,0]

#%%
oClass = base_model.layers[-6].output  # global mapping 2D layer
print(oClass.name)
mClass = keras.Model(model.input, outputs=oClass)

cw = model.get_layer('dense').kernel.numpy()[:,0]

#%% Fine tuning
base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
  
model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()
print(len(model.trainable_variables))

#%% Continue training
fine_tune_epochs = 2
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_batches,
                         epochs=total_epochs,
                         initial_epoch =  history.epoch[-1],
                         validation_data=validation_batches)

#%% Evaluate
loss2,accuracy2 = model.evaluate(validation_batches, steps = validation_steps)
print("final loss: {:.2f}".format(loss2))
print("final accuracy: {:.2f}".format(accuracy2))

#%% Fine tuning analysis
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

#%% Plot learning
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#%% Explore results
outs1 = base_model(image_batch).numpy()
cat = np.rollaxis(outs1[0,...], axis=2)
dog = np.rollaxis(outs1[3,...], axis=2)

oCNN = base_model.layers[12].output  # output of dense features
print(oCNN.name)
mCNN = keras.Model(base_model.input, outputs=oCNN)
outs2 = mCNN(image_batch).numpy()

lnames = [layer.name for layer in base_model.layers]
lints = [name for name in lnames if name.endswith('expand_relu')]
oall = [keras.Model(base_model.input, outputs=base_model.get_layer(name).output)(image_batch).numpy() for name in lints]

#%% Resize plot to image size
def resize_plot(plot_to_rescale, size) :
    return cv2.resize(plot_to_rescale, dsize=(size,size), interpolation=cv2.INTER_NEAREST)

#%% Plot CNN outputs
outs = outs2
rect = (5, 10)
ex_index = 2
CNN_index_start = 20
example = np.rollaxis(outs[ex_index,...], axis=2)

plt.subplot(1, 2, 1)
for i in range(rect[0] * rect[1]):
    plt.subplot(rect[0],rect[1],i+1)
    plt.xticks([])
    plt.yticks([])
    plt.title(i + CNN_index_start, fontsize=9)
    plt.imshow(example[i + CNN_index_start])

plt.subplot(1, 2, 2)
plt.title("Image: " + str(ex_index))
show_pic(image_batch[ex_index])

plt.show()

#%% Plot CNN outputs overlayed on image
CNN_index = 50
ex_index = 5

rect = (4, 4)
example = np.rollaxis(outs[ex_index,...], axis=2)

plt.figsize = (16,16)
for i in range(rect[0] * rect[1]):
    plt.subplot(rect[0],rect[1],i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow( unformat_pic(image_batch[ex_index]) )
    plt.imshow( resize_plot(example[CNN_index + i], IMG_SIZE), alpha=.5 )
    plt.title(CNN_index + i, fontsize=9)

plt.show()

#%% Plot averaged CNN outputs over image
for i in range(10) :
    acts = np.rollaxis(outs[i,...], axis=2)
    acts_avg = np.mean(acts, axis=0)
    plt.subplot(1,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow( unformat_pic(image_batch[i]) )
    plt.subplot(1,2,2)
    plt.imshow( resize_plot(acts_avg, IMG_SIZE))
    plt.suptitle("Image : {}".format(i))
    plt.xticks([])
    plt.yticks([])
    plt.show()

#%% All CNNs avgs for all layers
for i in range(10) :
    ex_index = i
    rect = (4, 4)
    
    oall_avg = [np.mean(oall[i], axis=-1) for i in range(len(oall))]
    acts_per_layer = [oall_avg[i][ex_index,...] for i in range(len(oall_avg))]
    
    plt.imshow( unformat_pic(image_batch[ex_index]) )
    plt.xticks([])
    plt.yticks([])
    plt.title("Image: {}".format(ex_index))
    plt.show()
    
    plt.figsize = (16,16)
    for i in range(rect[0] * rect[1]):
        plt.subplot(rect[0],rect[1],i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(acts_per_layer[i])
        # plt.imshow( resize_plot(example[CNN_index + i], IMG_SIZE))
        plt.title(i, fontsize=9)
    plt.show()


