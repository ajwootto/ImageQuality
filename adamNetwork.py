from keras.models import Sequential
from keras.models import model_from_json

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from scipy import misc
from keras.regularizers import l2, activity_l2

import scipy.io as sio
import numpy as np


img_dim = 50
num_train_samples = 800
num_test_samples = 200
num_classes = 2

start_image = 3

#Mat array format is weird, and flatten doesn't seem to do anything. Massage it into a 1D array
def massage_mat_array(array):
  names = np.zeros((len(array),), dtype='<U8')
  for i, name in enumerate(array):
    names[i] = name[0][0]
  return names

#method to shuffle two arrays the same way
def shuffle_in_unison(a, b):
  assert len(a) == len(b)
  shuffled_a = np.empty(a.shape, dtype=a.dtype)
  shuffled_b = np.empty(b.shape, dtype=b.dtype)
  permutation = np.random.permutation(len(a))
  for old_index, new_index in enumerate(permutation):
      shuffled_a[new_index] = a[old_index]
      shuffled_b[new_index] = b[old_index]
  return shuffled_a, shuffled_b

#load mat arrays with image filenames and corresponding scores
image_names = sio.loadmat('Data/AllImages_release.mat')['AllImages_release']
image_scores = sio.loadmat('Data/AllMOS_release.mat')['AllMOS_release'][0]

#massage the names array to be 1D
image_names = massage_mat_array(image_names)

image_names, image_scores = shuffle_in_unison(image_names, image_scores)

#define zero arrays for training and testing data
X_train = np.zeros((num_train_samples, 3, img_dim, img_dim), dtype='uint8')
Y_train = np.zeros((num_train_samples,), dtype='uint8')

X_test = np.zeros((num_test_samples, 3, img_dim, img_dim), dtype='uint8')
y_test = np.zeros((num_test_samples,), dtype='uint8')

#load training images into memory
for i in range(0, num_train_samples):
  img = misc.imread('Images/' + image_names[i])
  img = misc.imresize(img, (img_dim, img_dim))
  img = img.transpose((2,0,1))
  X_train[i, :, :, :] = img
  Y_train[i] = image_scores[i] #1 if (image_scores[i] > 50) else 0

#load testing images into memory
for i in range(0, num_test_samples):
  #start at index after the last index used by train samples
  test_index = i + num_train_samples
  img = misc.imread('Images/' + image_names[test_index])
  img = misc.imresize(img, (img_dim, img_dim))
  img = img.transpose((2,0,1))
  X_test[i, :, :, :] = img
  y_test[i] = image_scores[test_index] #1 if (image_scores[test_index] > 50) else 0

Y_train = np.reshape(Y_train, (len(Y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

#convert to categorical data type (not sure what this does yet but it was in the example)
#Y_train = np_utils.to_categorical(Y_train, num_classes)
#Y_test = np_utils.to_categorical(y_test, num_classes)
Y_test = y_test
#normalize image data to be between 0 and 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


def initialize_model():
  #create new model
  model = Sequential()

  #add layers
  #dim ordering parameter sets channel as the third dimension instead of the first (mirroring the way imread works)
  model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(img_dim, img_dim, 3), dim_ordering="tf"))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Dense(output_dim=num_classes))
  model.add(Activation("softmax"))

  return model

def train_model(model):
  #define optimizer with learning rate, momentum etc.
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  #model.compile(loss='categorical_crossentropy', optimizer=sgd)
  model.compile(loss='mean_squared_error', optimizer='rmsprop')
  #train model
  model.fit(X_train, Y_train,
                batch_size=5,
                nb_epoch=200,
                validation_data=(X_test, Y_test),
                shuffle=True)
  return model

def save_model(model):
  model_json = model.to_json()

  model_out = open('model.json', 'w')
  model_out.write(model_json)
  model_out.close()

  model.save_weights('weights')

def load_model():
  model_file = open('model.json', 'r')
  model_json = model_file.read()
  loaded_model = model_from_json(model_json)

  loaded_model.load_weights('weights')
  return loaded_model

#initializes a model based on the VGG architecture which placed second in 2014 imagenet competition
def initialize_vgg():
  model = Sequential()
  model.add(ZeroPadding2D((1,1), input_shape=(3,224,224)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(128, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(256, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1,1)))
  model.add(Convolution2D(512, 3, 3, activation='relu'))
  model.add(MaxPooling2D((2,2), strides=(2,2)))

  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='linear'))

  return model


#initialize a model based on the LeNet-5 architecture (1995)
def initialize_lenet():
  model = Sequential()
 
  # declare the first layers of convolution and pooling
  #
  num_filters = 6
  # side length of maxpooling square
  num_pool = 2
  # side length of convolution square
  num_conv = 5

  model.add(Convolution2D(6, num_conv, num_conv, border_mode='same', input_shape=(3, img_dim, img_dim)))
  model.add(Activation('relu'))
  #model.add(Convolution2D(num_filters, num_conv, num_conv))
  #model.add(Activation('relu'))
  model.add(MaxPooling2D((num_pool,num_pool)))
  model.add(Dropout(0.5))
   
  model.add(Convolution2D(16, num_conv, num_conv, border_mode='same' ))
  model.add(Activation('relu'))
  #model.add(Convolution2D(num_filters, num_conv, num_conv))
  #model.add(Activation('relu'))
  model.add(MaxPooling2D((num_pool,num_pool)))
  model.add(Dropout(0.5))
  
  model.add(Convolution2D(120, num_conv, num_conv))
  model.add(Activation('relu'))

  model.add(Flatten())
  model.add(Dense(84))  
                                   
  model.add(Activation('relu'))
  model.add(Dropout(0.5))           
   
  model.add(Dense(1, W_regularizer=l2(0.01) ))
  model.add(Activation('linear') )

  return model

model = initialize_lenet()
model = train_model(model)
save_model(model)

#output predicted classes of test data
predictions = model.predict_proba(X_test, batch_size=3, verbose=1)
print predictions
misclassified = np.sum(np.absolute(np.subtract(predictions, y_test.flatten())))

print misclassified
print float(misclassified) / float(num_test_samples) * 100.0



