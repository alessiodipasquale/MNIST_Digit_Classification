import numpy as np
import mnist
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

import mnist

num_filters = 8
filter_size = 3
pool_size = 2

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images =  mnist.test_images()
test_labels =  mnist.test_labels()


print(train_images.shape) #(60000,28,28)
print(train_labels.shape) #(60000,)

#Normalizziamo i valori dei pixel da [0,255] a [-0.5,0.5] e facciamo reshape di ogni immagine a 3 dimensioni (requisito di Keras)
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

print(train_images.shape) #(60000,28,28,1)
print(test_images.shape) #(10000,28,28,1)

model = keras.Sequential([
    Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=pool_size),
    Flatten(),
    Dense(10, activation='softmax'),
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=3,
  validation_data=(test_images, to_categorical(test_labels)),
)

model.save_weights('cnn.h5')
