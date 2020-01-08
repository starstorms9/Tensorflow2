#%% Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%% Common sets of ops
# In the tf.keras.layers package, layers are objects. To construct a layer,
# simply construct the object. Most layers take as a first argument the number
# of output dimensions / channels.
layer = tf.keras.layers.Dense(5)
# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to
# specify it manually, which is useful in some complex models.
layer = tf.keras.layers.Dense(3, input_shape=(None, 5))

# To use a layer, simply call it.
layer(tf.zeros([10, 5]))

# Layers have many useful methods. For example, you can inspect all variables
# in a layer using `layer.variables` and trainable variables using
# `layer.trainable_variables`. In this case a fully-connected layer
# will have variables for weights and biases.
layer.variables

# The variables are also accessible through nice accessors
layer.kernel, layer.bias

#%% Custom layer
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)

#%% Custom layer tests
layer = MyDenseLayer(3)
layer(tf.zeros([4, 5])) # Calling the layer `.builds` it.

print([var.name for var in layer.trainable_variables])

#%% Resnet block example class
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)

#%% Resnet testing
block = ResnetIdentityBlock(1, [1, 2, 3])
block(tf.zeros([1, 2, 3, 3])) 
block.layers
len(block.variables)
block.summary()

#%% Composing layers
my_seq = tf.keras.Sequential()
my_seq.add(tf.keras.layers.Conv2D(1, (1, 1), input_shape=(None, None, 3)))
my_seq.add(tf.keras.layers.BatchNormalization())
my_seq.add(tf.keras.layers.Conv2D(2, 1, padding='same'))
my_seq.add(tf.keras.layers.BatchNormalization())
my_seq.add(tf.keras.layers.Conv2D(3, (1, 1)))
my_seq.add(tf.keras.layers.BatchNormalization())

#%% Testing
my_seq(tf.zeros([1, 2, 3, 3]))
my_seq.summary()
