# standard imports
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine Learning Imports
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import (
    Dense, 
    Dropout, 
    Flatten,
    Conv2D,
    MaxPooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import scipy.misc
from skimage import transform
import warnings

warnings.filterwarnings("ignore")

# Sorting training data
train_direc = 'C://Users//leegu//OneDrive//Desktop//91319_211650_bundle_archive'
train = os.listdir(direc)

records = []
for category in train_list:
    if "nonsegmented" not in category:
        # append data
        img_list = os.listdir(direc + "//" + category)
        for img in img_list:
            records.append((img,category))
        
df_train = pd.DataFrame.from_records(records,columns=['image','category'])

# Sorting testing data
test_direc = 'C://Users//leegu//OneDrive//Desktop//91319_211650_bundle_archive//nonsegmentedv2'
test = os.listdir(test_direc)

import cv2

# Denoising
dim_image = []
for i in (train_direc + '//' + df_train['category'] + '//' + df_train['image']):
    if "nonsegmented" not in df_train['category']:
        im = cv2.imread(i)
        morph = im.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
        image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

        channel_height, channel_width, _ = image_channels[0].shape
        for i in range(0, 3):
            _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

        image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
        cv2.imwrite(i, image_channels)
        data = img.size
        dim_image.append(data[0])

i_height, i_width = min(dim_image), min(dim_image)

X = []
count = 0
bad_images = []
for i in (train_dir + "//" + df_train['category'] + '/' + df_train['image']):
    img = Image.open(i)
    img.load()
    img = np.asarray(img, dtype='float32')
    img = img/255
    data = transform.resize(img,(49,49))
    if data.size != 7203:
        bad_images.append(count)
    count += 1

df_train = df_train.drop(df_train.index[bad_images])
for i in (train_dir + df_train['category'] + '/' + df_train['image']):
    img = Image.open(i)
    img.load()
    img = np.asarray(img, dtype='float32')
    img = img/255
    data = transform.resize(img,(49,49))
    X.append(data)

X = np.array(X)
y = np.array(df_train['category'].astype('category').cat.codes)

from efficientnet import (
    EfficientNetB0,
    center_crop_and_resize, 
    preprocess_input
)


input_shape = (49,49,3)
batch_size = 10

# https://www.learnopencv.com/efficientnet-theory-code/
model = Sequential()

efficient_net = efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
model.add(efficient_net)
model.add(GlobalMaxPooling2D())
model.add(Dense(100, activation='relu'))
model.add(Flatten())
model.add(Dense(50,activation='relu'))
model.add(Flatten())
model.add(Dense(25,activation='relu'))
model.add(Flatten())
model.add(Dense(12,activation='softmax'))

model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.0001),
            metrics=['accuracy'])

model.fit(X_train, 
          y_train, 
          batch_size=batch_size, 
          epochs=10, 
          verbose=1, 
          validation_data=(X_test,y_test))

# Make predictions
model = EfficientNetB0(weights='imagenet')
image_size = model.input_shape[1]
x = center_crop_and_resize(image, image_size=image_size)
x = preprocess_input(x)
x = np.expand_dims(x, 0)

# make prediction and decode
y = model.predict(x)
decode_predictions(y)