#Test accuracy 63%

!pip install tensorflow
!pip install tqdm

import os
import cv2 as cv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform

DIR='/kaggle/input/chest-xray-pneumonia/chest_xray/train'

def extract_data(dir_path):
  classes=os.listdir(dir_path)
  Features=[]
  Labels=[]

  for c in classes:
    path= os.path.join(dir_path,c)
    images= os.listdir(path)
    print(f"Class ----------> {c}")
    for idx ,image in tqdm(enumerate(images) ,total=len(images)):
      image_path=os.path.join(path,image)
      img= cv.imread(image_path)
      # gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
      image = cv.resize(img, (64, 64))
      Features.append(image)
      Labels.append(c)


  Features=np.array(Features)
  # Labels=to_catagorical(Labels,num_classes= len(classes))
  return Labels , Features ,classes

# from google.colab import drive
# drive.mount('/content/drive')

Y_train, X_train ,classes = extract_data(DIR)
DIR_ = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'
Y_test, X_test , classes=extract_data(DIR_)

RES=[]
for i in Y_train:
  if i == "PNEUMONIA":
    RES.append([0,0])
  else:
    RES.append([1,0])

RES=np.array(RES)


Y_train=RES

REST=[]
for i in Y_test:
  if i == "PNEUMONIA":
    REST.append([0,0])
  else:
    REST.append([1,0])

REST=np.array(REST)


Y_test=REST

print(" X-train",X_train.shape)
print(" Y-train",Y_train.shape)
print(" X-test",X_test.shape)
print(" Y-test",Y_test.shape)

def identity_block_last(X,Y, f, filters, stage, block):

  '''
  Implementation of identity block described above

  Arguments:
  X -       input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
  f -       defines shpae of filter in the middle layer of the main path
  filters - list of integers, defining the number of filters in each layer of the main path
  stage -   defines the block position in the network
  block -   used for naming convention

  Returns:
  X - output is a tensor of shape (n_H, n_W, n_C) which matches (m, n_H_prev, n_W_prev, n_C_prev)
  '''

  # defining base name for block
  conv_base_name = 'res' + str(stage) + block + '_'
  bn_base_name = 'bn' + str(stage) + block + '_'

  # retrieve number of filters in each layer of main path
  # NOTE: f3 must be equal to n_C. That way dimensions of the third component will match the dimension of original input to identity block
  f1, f2, f3 = filters

  # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
  bn_axis = 3

  # save input for "addition" to last layer output; step in skip-connection
  X_skip_connection = X

  # ----------------------------------------------------------------------
  # Building layers/component of identity block using Keras functional API

  # First component/layer of main path
  X = Conv2D(filters= f1, kernel_size = (1,1), strides = (1,1), padding='valid', name=conv_base_name+'first_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'first_component')(X)
  X = Activation('relu')(X)

  # Second component/layer of main path
  X = Conv2D(filters= f2, kernel_size = (f,f), strides = (1,1), padding='same', name=conv_base_name+'second_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'second_component')(X)
  X = Activation('relu')(X)

  Y = Conv2D(f2, (1, 1), strides = (1,1), padding = 'same', name = conv_base_name + 'merge', kernel_initializer = glorot_uniform(seed=0))(Y)
  Y = BatchNormalization(axis = 3, name = bn_base_name + 'merge')(Y)

  X = Add()([X,Y])
  X = Activation('relu')(X)

  # Third component/layer of main path
  X = Conv2D(filters= f3, kernel_size = (1,1), strides = (1,1), padding='valid', name=conv_base_name+'third_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'third_component')(X)
    
  # "Addition step" - skip-connection value merges with main path
  # NOTE: both values have same dimensions at this point, so no operation is required to match dimensions
  X = Add()([X, X_skip_connection])
  X = Activation('relu')(X)

  return X


def identity_block(X,Y, f, filters, stage, block):

  '''
  Implementation of identity block described above

  Arguments:
  X -       input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
  f -       defines shpae of filter in the middle layer of the main path
  filters - list of integers, defining the number of filters in each layer of the main path
  stage -   defines the block position in the network
  block -   used for naming convention

  Returns:
  X - output is a tensor of shape (n_H, n_W, n_C) which matches (m, n_H_prev, n_W_prev, n_C_prev)
  '''

  # defining base name for block
  conv_base_name = 'res' + str(stage) + block + '_'
  bn_base_name = 'bn' + str(stage) + block + '_'

  # retrieve number of filters in each layer of main path
  # NOTE: f3 must be equal to n_C. That way dimensions of the third component will match the dimension of original input to identity block
  f1, f2, f3 = filters

  # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
  bn_axis = 3

  # save input for "addition" to last layer output; step in skip-connection
  X_skip_connection = X

  # ----------------------------------------------------------------------
  # Building layers/component of identity block using Keras functional API

  # First component/layer of main path
  X = Conv2D(filters= f1, kernel_size = (1,1), strides = (1,1), padding='valid', name=conv_base_name+'first_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'first_component')(X)
  X = Activation('relu')(X)

  # Second component/layer of main path
  X = Conv2D(filters= f2, kernel_size = (f,f), strides = (1,1), padding='same', name=conv_base_name+'second_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'second_component')(X)
  X = Activation('relu')(X)
  
  Y = Conv2D(f2, (1, 1), strides = (1,1), padding = 'same', name = conv_base_name + 'merge', kernel_initializer = glorot_uniform(seed=0))(Y)
  Y = BatchNormalization(axis = 3, name = bn_base_name + 'merge')(Y)

  X = Add()([X,Y])
  X = Activation('relu')(X)

  Y = X

  # Third component/layer of main path
  X = Conv2D(filters= f3, kernel_size = (1,1), strides = (1,1), padding='valid', name=conv_base_name+'third_component', kernel_initializer = glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=bn_axis, name=bn_base_name+'third_component')(X)

  # "Addition step" - skip-connection value merges with main path
  # NOTE: both values have same dimensions at this point, so no operation is required to match dimensions
  X = Add()([X, X_skip_connection])
  X = Activation('relu')(X)

  return X , Y


def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in above figure

    Arguments:
    X -       input tensor to the block of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -       defines shape of filter in the middle layer of the main path
    filters - list of integers, defining the number of filters in each layer of the main path
    stage -   defines the block position in the network
    block -   used for naming convention
    s -       specifies the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining base name for block
    conv_base_name = 'res' + str(stage) + block + '_'
    bn_base_name = 'bn' + str(stage) + block + '_'

    # retrieve number of filters in each layer of main path
    f1, f2, f3 = filters

    # Batch normalization must be performed on the 'channels' axis for input. It is 3, for our case
    bn_axis = 3

    # save input for "addition" to last layer output; step in skip-connection
    X_skip_connection = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(f1, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'first_component', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = bn_axis, name = bn_base_name + 'first_component')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(f2,  kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_base_name + 'second_component', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = bn_axis, name = bn_base_name + 'second_component')(X)
    X = Activation('relu')(X)
    
    Y = X

    # Third component of main path
    X = Conv2D(f3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_base_name + 'third_component', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = bn_axis, name = bn_base_name + 'third_component')(X)

    ##### Convolve skip-connection value to match its dimensions to third layer output's dimensions ####
    X_skip_connection = Conv2D(f3, (1, 1), strides = (s,s), padding = 'valid', name = conv_base_name + 'merge', kernel_initializer = glorot_uniform(seed=0))(X_skip_connection)
    X_skip_connection = BatchNormalization(axis = 3, name = bn_base_name + 'merge')(X_skip_connection)

    # "Addition step"
    # NOTE: both values have same dimensions at this point
    X = Add()([X, X_skip_connection])
    X = Activation('relu')(X)

    return X , Y


def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Arguments:
    input_shape - shape of the images of the dataset
    classes - number of classes

    Returns:
    model - a Model() instance in Keras

    """

    # plug in input_shape to define the input tensor
    X_input = Input(input_shape)

    # Zero-Padding : pads the input with a pad of (3,3)
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv_1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # NOTE: dimensions of filters that are passed to identity block are such that final layer output
    # in identity block mathces the original input to the block
    # blocks in each stage are alphabetically sequenced
    
    Y = X

    # Stage 2
    X , Y = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X , Y= identity_block(X,Y, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block_last(X,Y, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X , Y= convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X , Y= identity_block(X,Y, 3, [128, 128, 512], stage=3, block='b')
    X , Y= identity_block(X,Y, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block_last(X,Y, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X , Y= convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X , Y= identity_block(X,Y, 3, [256, 256, 1024], stage=4, block='b')
    X , Y= identity_block(X,Y, 3, [256, 256, 1024], stage=4, block='c')
    X , Y= identity_block(X,Y, 3, [256, 256, 1024], stage=4, block='d')
    X , Y= identity_block(X,Y, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block_last(X,Y, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X , Y= convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X , Y= identity_block(X,Y, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block_last(X,Y, 3, [512, 512, 2048], stage=5, block='c')

    # Average Pooling
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

model = ResNet50(input_shape = (64, 64, 3), classes = 2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs = 10, batch_size = 32)

predictions = model.evaluate(X_test, Y_test)
print("Loss = " + str(predictions[0]))
print("Test Accuracy = " + str(predictions[1]))
