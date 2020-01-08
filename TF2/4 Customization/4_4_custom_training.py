#%% Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#%% TF variables
# Using Python state
x = tf.zeros([10, 10])
x += 2  # This is equivalent to x = x + 2, which does not mutate the original value of x
print(x)

v = tf.Variable(1.0)
# Use Python's `assert` as a debugging statement to test the condition
assert v.numpy() == 1.0

# Reassign the value `v`
v.assign(3.0)
assert v.numpy() == 3.0

# Use `v` in a TensorFlow `tf.square()` operation and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0

#%% Building a model class
class Model(object):
  def __init__(self, W_init=5.0, b_init=0.0):
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    self.W = tf.Variable(W_init)
    self.b = tf.Variable(b_init)

  def __call__(self, x):
    return self.W * x + self.b

def loss(predicted_y, target_y):
  return tf.reduce_mean(tf.square(predicted_y - target_y))

#%% Test model
model = Model()
assert model(3.0).numpy() == 15.0

#%% Gen training data
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 100

inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise * 1

#%% Graph data
plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss: %1.6f' % loss(model(inputs), outputs).numpy())

#%% Training loop
def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)
  dW, db = t.gradient(current_loss, [model.W, model.b])
  model.W.assign_sub(learning_rate * dW)
  model.b.assign_sub(learning_rate * db)
  
#%% Make model
model = Model(5.0, 0.1)

# Train model
# Collect the history of W-values and b-values to plot later
Ws, bs, lh = [], [], []
epochs = range(10)
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(model(inputs), outputs)
  lh.append(current_loss)

  train(model, inputs, outputs, learning_rate=0.1)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

#%% Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b',
         epochs, lh, 'g')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'True W', 'True b'])
plt.show()