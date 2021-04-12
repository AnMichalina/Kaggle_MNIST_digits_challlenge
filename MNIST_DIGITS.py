from keras import Input, Model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.models import Sequential
import random
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Conv2D, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import math
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras import layers
import warnings

warnings.filterwarnings("ignore")

train_path = "data//train.csv"
test_path = "data//test.csv"
df_train = pd.read_csv(train_path)
X_test = pd.read_csv(test_path)

X_train1 = df_train.drop(['label'], axis=1)
y_train = df_train['label']

print('Dimension of training images:', np.shape(X_train1))
print('Dimension of trainig labels:', np.shape(y_train))
print('Dimension of testing images:', np.shape(X_test))

X_train1 = X_train1.astype('float64')
X_test = X_test.astype('float64')
X_train1 /= 255
X_test /= 255

X_train1 = np.asarray(X_train1)
X_test = np.asarray(X_test)

X_train1 = X_train1.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

print('Dimension of training images:', np.shape(X_train1))
print('Dimension of testing images:', np.shape(X_test))

# def preprocess_image(image, distort=True):
#     if distort:
#         # Randomly flip the image horizontally.
#         image = tf.image.random_flip_left_right(image)
#         image = tf.image.random_brightness(image, max_delta=63)
#         image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
#
#         rotate_pct = 0.5  # 50% of the time do a rotation between 0 to 90 degrees
#         if random.random() < rotate_pct:
#             degrees = random.randint(0, 90)
#             image = tfa.image.rotate(image, degrees * math.pi / 180, interpolation='BILINEAR')
#
#         # Fixed standardization
#         image = (tf.cast(image, tf.float32) - 127.5) / 128.0
#         # Subtract off the mean and divide by the variance of the pixels.
#
#     image = tf.image.per_image_standardization(image)
#
#     return image


X_train = X_train1

# pick a sample to plot
# for sample in range(10):
#     image = X_train[sample]
#     plt.title(" Digit " + str(y_train[sample]))
#     # plot the sample
#     fig = plt.figure
#     plt.imshow(image, cmap='gray')
#     plt.show()

y_train = np_utils.to_categorical(y_train)
# print(y_train[0])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2)

print('Build model...')

# model = keras.Sequential([
#     layers.Dense(784, activation='relu'),
#     layers.Dropout(0.3),
#     layers.BatchNormalization(),
#     layers.Dense(10, activation="softmax")
# ])

model = keras.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2), strides=(2, 2)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation="softmax")
])
# dummy_dat = np.zeros((37800, 28, 28, 1), dtype=np.float64)
# fudge_X_train = np.concatenate((X_train, dummy_dat), axis=3)
fudge_X_train = X_train
fudge_y_train = y_train

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(fudge_X_train)

a = np.concatenate((X_train, fudge_X_train), axis=0)
b = np.concatenate((y_train, fudge_y_train), axis=0)
c = np.concatenate((a, b), axis=1)
print (c)

# x_batches = fudge_X_train
# y_batches = y_train

# epochs = 5
#
# for e in range(epochs):
#     #print('Epoch', e)
#     batches = 0
#     per_batch = 5
#     for x_batch, y_batch in datagen.flow(fudge_X_train, y_train, batch_size=per_batch):
#         x_batches = np.concatenate((x_batches, x_batch), axis=0)
#         y_batches = np.concatenate((y_batches, y_batch), axis=0)
#         batches += 1
#         if batches >= len(fudge_X_train) / per_batch:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break
#
# X_train_new = x_batches[:, :, :, :2]
# print(X_train_new.shape)
# print(y_batches.shape)

# Insert Hyperparameters
# learning_rate = 0.1
training_epochs = 20
batch_size = 100
# sgd = tf.optimizers.SGD(lr=learning_rate)

# We rely on the plain vanilla Stochastic Gradient Descent as our optimizing methodology
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

# X_train = X_train.reshape((-1, 784))
# X_val = X_val.reshape((-1, 784))

# history = model.fit(X_train_new, y_train,
#                     batch_size=batch_size,
#                     epochs=training_epochs,
#                     verbose=2,
#                     validation_data=(X_val, y_val))

history = model.fit(datagen.flow(X_train, y_train, batch_size = batch_size),
 validation_data = (X_val, y_val), steps_per_epoch = len(X_train) // 100,
 epochs = training_epochs)

model.summary()  # We have 297,910 parameters to estimate

#
# model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same',
#                  activation ='relu', input_shape = (28,28,1)))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# #
# model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same',
#                  activation ='relu'))
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.25))
# # fully connected
# model.add(Flatten())
# model.add(Dense(256, activation = "relu"))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation = "softmax"))
#
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#
# epochs = 10  # for better result increase the epochs
# batch_size = 250


# data augmentation
# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # dimesion reduction
#         rotation_range=5,  # randomly rotate images in the range 5 degrees
#         zoom_range = 0.1, # Randomly zoom image 10%
#         width_shift_range=0.1,  # randomly shift images horizontally 10%
#         height_shift_range=0.1,  # randomly shift images vertically 10%
#         horizontal_flip=False,  # randomly flip images
#         vertical_flip=False)  # randomly flip images
#
# datagen.fit(X_train)
#
# # Fit the model
# history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
#                               epochs = epochs, validation_data = (X_val,y_val), steps_per_epoch=X_train.shape[0] // batch_size)

# X_test = X_test.reshape((-1, 784))

test_pred = pd.DataFrame(model.predict(X_test))
test_pred = pd.DataFrame(test_pred.idxmax(axis=1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns={0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

print(test_pred.head())

test_pred.to_csv('mnist_submission.csv', index=False)
