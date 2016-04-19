from PIL import Image
import numpy as np
import glob
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

test_folder_path = './LIVE Dataset/Images/testing/'
test_bmps_path = glob.glob(test_folder_path+'/*.bmp') 

test_bmps = np.array([np.array(Image.open(test_bmps_path[i])) for i in range(len(test_bmps_path))], dtype='float32') / 255
test_bmps = test_bmps.transpose(0,3,1,2)
print 'images read: '
print test_bmps.shape[0]

test_scores = np.array([63.96340,25.33530,48.93656,35.88633,66.50923,54.57970,77.88214], dtype='float32')
test_scores = test_scores/50 - 1

model = Sequential()
model.add(Convolution2D(10, 5, 5, input_shape=(3, 500, 500)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(4, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')
print 'model compiled'

model.fit(test_bmps, test_scores, nb_epoch=20)
print 'model fit'

predictions = model.predict_proba(test_bmps)
print 'model predictions: '
print predictions

print 'real answers: '
print test_scores