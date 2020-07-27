# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:17:15 2020

@author: ASUS
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
model = load_model("giaydep_classify3.h5")
model.summary()
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#%%
from keras.preprocessing import image
test_image= image.load_img(r"D:\Hahalolo\persona\test_images\5.jpg") 
test_image = image.img_to_array(test_image)
test_image = tf.image.rgb_to_grayscale(test_image, name=test_image)
test_image = tf.image.resize(test_image, (28,28))
test_image = test_image / 255.0
#test_image = Image.fromarray(test_image).resize(size=(28, 28))
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)   
print(result.max())
#%% check keras version
import h5py

f = h5py.File('giaydep_classify3.h5', 'r')
print(f.attrs.get('keras_version'))

import keras
print(keras.__version__)
