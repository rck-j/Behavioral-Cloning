import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('../a/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

all_images = []
steering_angle = []
for line in lines:
    center = line[0]
    left = line[1]
    right = line[2]
    center_t = center.split('/')
    left_t = left.split('/')
    right_t = right.split('/')
    center_f = center_t[-1]
    left_f = left_t[-1]
    right_f = right_t[-1]
    center_p = '../a/IMG/' + center_f
    right_p = '../a/IMG/' + right_f
    left_p = '../a/IMG/' + left_f
    center_i = cv2.imread(center_p)
    left_i = cv2.imread(left_p)
    right_i = cv2.imread(right_p)
    all_images.append(center_i)
    all_images.append(left_i)
    all_images.append(right_i)
    angle = float(line[3])
    steering_angle.append(angle)
    steering_angle.append(angle + 0.2)
    steering_angle.append(angle - 0.2)


# 
# def PreProc(images):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
#     for i in range(images.shape[0]):
#         images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2YUV)
#         images[i][:,:,0] = clahe.apply(images[i][:,:,0])
#         images[i] = cv2.cvtColor(images[i], cv2.COLOR_YUV2RGB)
#
#     return images


print(len(all_images))
print(len(steering_angle))

cropped = [img[56:160, 0:320] for img in all_images]
resized = [cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA) for crop in cropped]
X_train = np.array(resized)
y_train = np.array(steering_angle)

print(X_train[0].shape)
print(y_train.shape)
i = np.copy(X_train[0])

#X_train = PreProc(X_train)

j = X_train[0]
cv2.imwrite('./test.jpg', i)
cv2.imwrite('./test1.jpg', j)


model = Sequential()
model.add(Lambda(lambda x: (x /255.0) - 0.5, input_shape=(32, 32, 3)))

model.add(Convolution2D(24, 5, 5, border_mode='same', activation='elu'))
#model.add(Convolution2D(32, 3, 3, border_mode='same', activation='elu'))
#model.add(Convolution2D(32, 3, 3, border_mode='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', activation='elu'))
#model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
#model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))


model.add(Convolution2D(48, 5, 5, border_mode='same', activation='elu'))
#model.add(Convolution2D(128, 3, 3, border_mode='same', activation='elu'))
#model.add(Convolution2D(128, 3, 3, border_mode='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))


model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))
#model.add(Convolution2D(256, 3, 3, border_mode='same', activation='elu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))


##model.add(Convolution2D(256, 3, 3, activation='relu'))
##model.add(Convolution2D(256, 3, 3, activation='relu'))
##model.add(Convolution2D(256, 3, 3, activation='relu'))
##model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))

model.add(Flatten())
##model.add(Dense(4096))
##model.add(Dense(2048))
model.add(Dense(1024))
model.add(Dropout(.80))
model.add(Dense(512))
model.add(Dropout(.80))
model.add(Dense(256))
model.add(Dropout(.80))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))

model.add(Dense(1))

adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('./model.h5')
