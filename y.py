import os
import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Dropout, Merge
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

samples = []
with open('../a/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = '../a/IMG/'+batch_sample[i].split('/')[-1]
                    all_images = cv2.imread(name)
                    angle = float(batch_sample[3])
                    images.append(all_images)
                    angles.append(angle)

            # trim image to only see section with road
            cropped = [img[56:160, 0:320] for img in images]
            resized = [cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA) for crop in cropped]
            X_train = np.array(resized)
            y_train = np.array(angles)
            Z_train = np.copy(X_train)
            a_train = np.copy(y_train)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train1 = generator(train_samples, batch_size=32)
train2 = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 160, 320  # Trimmed image format

branch1 = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
#model.add(Lambda(lambda x: (x /255.0) - 0.5, input_shape=(160, 320, 3)))
branch1.add(Lambda(lambda x: (x /255.0) - 0.5, input_shape=(32, 32, 3)))
# model.add(Lambda(lambda x: x/127.5 - 1.,
#         input_shape=(ch, row, col),
#         output_shape=(ch, row, col)))
#model.add(... finish defining the rest of your model architecture here ...)
branch1.add(Convolution2D(2, 1, 1, border_mode='same', activation='elu'))
branch1.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
branch1.add(Convolution2D(4, 1, 1, border_mode='same', activation='elu'))
branch1.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
# model.add(Convolution2D(32, 1, 1, border_mode='same', activation='elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
# model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
# model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

# branch2 = Sequential()
# branch2.add(Lambda(lambda x: (x /255.0) - 0.5, input_shape=(32, 32, 3)))
# branch2.add(Convolution2D(1, 1, 1, border_mode='same', activation='elu'))
# branch2.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

# model = Sequential()
# model.add(Merge([branch1, branch2], mode='concat'))
branch1.add(Flatten())

# model.add(Dense(3200))
# model.add(Dropout(.80))
# model.add(Dense(1000))
# model.add(Dropout(.80))
branch1.add(Dense(100))
#model.add(Dropout(.80))
branch1.add(Dense(10))
#model.add(Dense(64))
#model.add(Dense(32))
#model.add(Dense(16))

branch1.add(Dense(1))
branch1.compile(loss='mse', optimizer='adam')
branch1.fit_generator(train1, samples_per_epoch=22080, validation_data=validation_generator, nb_val_samples=5520, nb_epoch=3)

branch1.save('./model.h5')
