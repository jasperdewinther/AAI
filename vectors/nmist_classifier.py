import pickle, gzip, os
from urllib import request
from pylab import imshow, show, cm
import numpy as np
import network
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def fixlabels(labels):
    new_labels = []
    for label in labels:
        tmp_arr = []
        for i in range(10):
            if i == label:
                tmp_arr.append(1)
            else:
                tmp_arr.append(0)
        new_labels.append(np.array(tmp_arr))

    return np.array(new_labels)

def get_image (number):
    (data, label) = [img[number] for img in train_set ]
    return (np.array(data), numberlabel_to_array(label))

def view_image (number):
    (X, y) = get_image (number)
    imshow (X.reshape(28 ,28), cmap=cm.gray)
    show()

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding ='latin1')
f.close()

train = [img for img in train_set][0]
train_labels = fixlabels(train_set[1])

validation = [img for img in valid_set][0]
validation_labels = fixlabels(valid_set[1])


#more nodes means higher accuracy, 16 seems to be pretty good accuracy/speed tradeoff
model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.sigmoid)
])

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])

epochs = 5

history = model.fit(x=train, y=train_labels, validation_data=(validation,validation_labels),epochs=epochs)

model.summary()

accuracy=history.history['acc']
val_accuracy = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()