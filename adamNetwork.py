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

import matplotlib.pyplot as plt
import math
import random
import os

img_dim = 50

num_classes = 2

start_image = 3

data_folders = ['Images', 'redditGood', 'bingBad']

reddit_photos = os.listdir('redditGood')
reddit_photos = [x for x in reddit_photos if '.jpg' in x]
random.shuffle(reddit_photos)

bing_photos = os.listdir('bingBad')
random.shuffle(bing_photos)

mode = 'regression'

if mode=='categorical':
  num_train_samples = 900
  num_test_samples = 300
elif mode =='regression':
  num_train_samples = 800
  num_test_samples = 200

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

def normalize_reddit_score(score):
  score = score + 70
  if score > 100:
    score = 100
  return score 

def load_image(name):
  img = misc.imread(name)
  img = misc.imresize(img, (img_dim, img_dim))
  img = img.transpose((2,0,1))
  return img


def load_images(num_from_each, start_index=0):
  num = num_from_each * 3
  X = np.zeros((num, 3, img_dim, img_dim), dtype='uint8')
  Y = np.zeros((num,), dtype='uint8')
  print "Processing Reddit Photos"
  for i in range(start_index, num_from_each + start_index):
    try:
      img = load_image('redditGood/' + reddit_photos[i])
    except:
      continue
    X[i-start_index, :, :, :] = img[0:3, :, :]
    Y[i-start_index] = 1 

  print "Processing Bing Photos"
  for i in range(start_index, num_from_each + start_index): 
      img = load_image('bingBad/' + bing_photos[i])
      X[i-start_index+num_from_each, :, :, :] = img[0:3, :, :]
      Y[i-start_index+num_from_each] = 0
  print "Processing LIVE Photos"
  for i in range(start_index, num_from_each + start_index):  
      img = load_image('Images/' + image_names[i])
      X[i-start_index+num_from_each*2, :, :, :] = img[0:3, :, :]
      Y[i-start_index+num_from_each*2] = 1 if image_scores[i] > 60 else 0

  return X, Y

def load_live_images(num, start_index=0):
  X = np.zeros((num, 3, img_dim, img_dim), dtype='uint8')
  Y = np.zeros((num,), dtype='uint8')
  print "Processing LIVE Photos"
  for i in range(start_index, num + start_index):  
      img = load_image('Images/' + image_names[i])
      X[i-start_index, :, :, :] = img[0:3, :, :]
      Y[i-start_index] = image_scores[i]
  return X, Y

#load training images into memory
if mode == 'regression':
  X_train, Y_train = load_live_images(num_train_samples)
  X_test, Y_test = load_live_images(num_test_samples, start_index=num_train_samples)
  Y_train = Y_train/50.0 - 1
  Y_test = Y_test/50.0 - 1
elif mode == 'categorical':
  X_train, Y_train = load_images(num_train_samples/3)
  X_test, Y_test = load_images(num_test_samples/3, start_index=num_train_samples/3)

Y_train = np.reshape(Y_train, (len(Y_train), 1))
Y_test = np.reshape(Y_test, (len(Y_test), 1))

print Y_test



#convert to categorical data type for cross entropy calculations
if mode == 'categorical':
  Y_train_cat = np_utils.to_categorical(Y_train, num_classes)
  Y_test_cat = np_utils.to_categorical(y_test, num_classes)

#normalize image data to be between 0 and 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


def initialize_model():
  #create new model
  model = Sequential()

  #add layers
  model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, img_dim, img_dim)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(512))
  model.add(Dense(output_dim=num_classes))
  model.add(Activation("softmax"))

  return model

def train_model_categorical(model):
  #define optimizer with learning rate, momentum etc.
  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd)
  #model.compile(loss='mean_squared_error', optimizer='rmsprop')
  #train model
  for i in range(0, 10):
    model.fit(X_train, Y_train_cat,
                  batch_size=32,
                  nb_epoch=100,
                  validation_data=(X_test, Y_test_cat),
                  shuffle=True)
    print "Saving"
    save_model(model, "categorical")
  return model

def train_model_regression(model):
  #define optimizer with learning rate, momentum etc.
  model.compile(loss='mean_squared_error', optimizer='rmsprop')
  #train model
  for i in range(0, 10):
    model.fit(X_train, Y_train,
                  batch_size=32,
                  nb_epoch=100,
                  validation_data=(X_test, Y_test),
                  shuffle=True)
    print "Saving"
    save_model(model, "regression")
  return model

def save_model(model, name="model"):
  model_json = model.to_json()

  model_out = open(name + '.json', 'w')
  model_out.write(model_json)
  model_out.close()

  model.save_weights('weights_' + name, overwrite=True)

def load_model(name):
  model_file = open(name + '.json', 'r')
  model_json = model_file.read()
  loaded_model = model_from_json(model_json)

  loaded_model.load_weights('weights_' + name)
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
  model.add(MaxPooling2D((num_pool,num_pool)))
  model.add(Convolution2D(16, num_conv, num_conv, border_mode='same' ))
  model.add(Activation('relu'))
  
  model.add(MaxPooling2D((num_pool,num_pool)))
  model.add(Convolution2D(120, num_conv, num_conv))
  model.add(Activation('relu'))

  model.add(Flatten())
  model.add(Dense(84))                           
  model.add(Activation('relu'))
   
  return model

def attach_regression_output(model):
  model.add(Dense(1, W_regularizer=l2(0.01) ))
  model.add(Activation('linear') )
  return model

def attach_class_output(model):
  model.add(Dense(output_dim=num_classes))
  model.add(Activation("softmax"))
  return model

def plot_weights(model):
  for k, layer in enumerate(model.layers):
    if layer.name != 'convolution2d':
      continue
    fig = plt.figure(figsize=(11,30))
    plt.title('Convolutional Layer ' + str(k))
    weights = layer.get_weights()[0]
    weights = np.divide(weights, weights.max())
    weights = np.multiply(weights, 255)
    index = 1
    for i, kernel in enumerate(weights):
      for j, channel in enumerate(kernel):
          a=fig.add_subplot(len(weights), len(kernel), index)
          plt.imshow(channel, cmap='Greys_r')
          plt.title('Filter ' + str(i+1) + ' Channel ' + str(j+1))
          index += 1
    plt.show()


model = initialize_lenet()
if mode == 'categorical':
  model = attach_class_output(model)
  model = train_model_categorical(model)
elif mode ==  'regression':
  model = attach_regression_output(model)
  model = train_model_regression(model)

#model = load_model()
# save_model(model)

#output predicted classes of test data
if mode == 'categorical':
  predictions = model.predict_classes(X_test, batch_size=3, verbose=1)
elif mode == 'regression':
  predictions = model.predict(X_test, batch_size=3, verbose=1)

print predictions
print np.absolute(np.subtract(predictions.flatten(), y_test.flatten()))
misclassified = np.sum(np.absolute(np.subtract(predictions.flatten(), y_test.flatten())))

print misclassified
print float(misclassified) / float(num_test_samples) 

#plot_weights(model)

#test a specific image against the trained model
def test_image_score(name):
  tests = np.zeros((1, 3, img_dim, img_dim), dtype='uint8')

  image = load_image(name)
  tests[0, :, :, :] = image

  tests = tests.astype('float32')
  tests /= 255
  print model.predict(tests)




