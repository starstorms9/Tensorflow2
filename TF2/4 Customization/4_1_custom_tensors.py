#%% Imports
import tensorflow as tf
import numpy as np
import time
import tempfile

#%% Tensors
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

#%% TF Matmul
x = tf.matmul([[1.0]], [[2.0, 3.0]])
print(x)
print(x.shape)
print(x.dtype)

#%% Numpy relation
ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

#%% GPU accel
x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.config.experimental.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

#%% Explicit device placement
def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)
  result = time.time()-start
  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.config.experimental.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)

# Compare to numpy
print("On CPU with Numpy:")
def time_matmul_np(x):
  start = time.time()
  for loop in range(10):
    np.matmul(x, x)
  result = time.time()-start
  print("10 loops: {:0.2f}ms".format(1000*result))

xnp = np.array(np.random.rand(1000,1000), dtype='f')
time_matmul_np(xnp)
    
#%% Dataset
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
            Line 2
            Line 3
              """)

ds_file = tf.data.TextLineDataset(filename)

#%% Transform numerical tensor
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
ds_tensors_sq = ds_tensors.map(tf.square).shuffle(2).batch(2)
print(list(ds_tensors_sq))

#%% Custom map function
def add_one_tf(x):
    return tf.add(x, 1)

ds_tensors_map = ds_tensors.map(add_one_tf)
print(list(ds_tensors_map))

# Using lambda
print(list(ds_tensors.map(lambda x: tf.add(x,488))))

#%% Transform text tensor
ds_file = ds_file.batch(2)
print(list(ds_file))

#%% Iterate
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)