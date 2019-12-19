import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import matplotlib.pyplot as plt
from itertools import cycle

import numpy as np
print(tf.__version__)
np.set_printoptions(precision=3, suppress=True)

#%% Download data
(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k', 
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure. 
    with_info=True)

#%% Encoder
encoder = info.features['text'].encoder
print ('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'movie great hate movie love director hazelnut'

encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))

assert original_string == sample_string

for ts in encoded_string:
  print ('{} ----> {}'.format(ts, encoder.decode([ts])))
  
#%% Explore data
trn_txt, trn_lab = None, None
for train_example, train_label in train_data.take(1):
  trn_txt, trn_lab = train_example, train_label
  print('Encoded text:', train_example[:20].numpy())
  print('Label:', train_label.numpy())
  
encoder.decode(train_example)

#%% Prep data for training
BUFFER_SIZE = 1000

train_batches = (train_data.shuffle(BUFFER_SIZE).padded_batch(32, train_data.output_shapes))
test_batches = (test_data.padded_batch(32, train_data.output_shapes))

for example_batch, label_batch in train_batches.take(2):
  print("Batch shape:", example_batch.shape)
  print("label shape:", label_batch.shape)
  
#%% Build model
model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 1),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1, activation='sigmoid')])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%% Train model
history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

#%% Evaluate
loss, accuracy = model.evaluate(test_batches)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

#%% Graph history
history_dict = history.history
keys = history_dict.keys()

acc, val_acc, loss, val_loss = [history.history[key] for key in history.history.keys()]
# acc = history_dict['accuracy']
# val_acc = history_dict['val_accuracy']
# loss = history_dict['loss']
# val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo-', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'bo--', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
plt.plot(epochs, acc, ':', label='Training acc')
plt.plot(epochs, val_acc, 'bo-', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

#%% Methods
def normalize_rows(x: np.ndarray):
    return x/np.linalg.norm(x, ord=2, axis=1, keepdims=True)

def Distance(a, b):
    return np.linalg.norm(a-b)

def get_Embed_Dist(txt1, txt2):
    encoded = np.array(encoder.encode(txt1 + txt2))
    embeds = lay_emb(encoded).numpy()
    return Distance(embeds[0], embeds[1])

#%% Model exploration
chktxt = ['hate', 'love', 'movie', 'director', 'horrible', 'experience', 'well', 'great', 'worst']
chkcode = np.array([encoder.encode(word)[0] for word in chktxt])
# chkcode = np.random.randint(0, 1000, size=4)
# chktxt = [ encoder.decode([word]) for word in list(chkcode)]

lay_emb = model.layers[0]
embeds = lay_emb(chkcode).numpy()
# embeds = normalize_rows(embeds)

lines = cycle(["-","--","-.",":"])

for i in range(len(embeds)):
    plt.plot(embeds[i], 'o' + next(lines), label=chktxt[i])

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

#%% Find best and worst words
beds = np.array(lay_emb.get_weights())[0,:,0]
wordmax, wordmin = np.argmax(beds), np.argmin(beds)

decoded = encoder.decode([wordmax, wordmin])