# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:02:46 2020

@author: ASUS
"""
import tensorflow as tf
import pathlib
dataset_url = "https://hoangnhi2310.github.io/giaydep2.tar"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='giaydep', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)
# image_count = len(list(data_dir.glob('*/*')))
# print(image_count)
#%% load from local
import tensorflow as tf
import pathlib
# path = "D:\hoangnhi2310.github.io\hoangnhi2310.github.io\giaydep2.tar"
path = r"D:\Hahalolo\persona\test_images\Giày dep\giaydep.tar"
data_dir = tf.keras.utils.get_file(origin=path, 
                                   fname='giaydep', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)
#%%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(300,300),
    batch_size=128,
    class_mode='binary'
)
#%%
import cv2
import glob
count = 1
path_dep = 'giay/*'
path_giay = 'dep/*'
paths_list = [path_dep, path_giay]
for path in paths_list:
    filenames = data_dir.glob(path)
    for filename in filenames:
      image = cv2.imread(filename)
      gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image = cv2.resize(image, (128, 128))
      gray_img = cv2.resize(gray_img, (128, 128))
      cv2.imwrite("gray_images/gray_" +str(count) +".jpg", gray_img)
      cv2.imwrite("color_images/color_" +str(count) +".jpg", image)
      count += 1
#%%
from keras.preprocessing import image
test_image= image.load_img(r"D:\Hahalolo\persona\test_images\5.jpg") 
test_image = image.img_to_array(test_image)
test_image = tf.image.rgb_to_grayscale(test_image, name=test_image)
#%%
path_local = r"giay\IMG_6338.JPG"
# filename = glob.glob('giay/IMG_6338.JPG')
image = cv2.imread(path_local, 0)
cv2.imshow('image', image)
image = cv2.imread(filename)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (128, 128))
gray_img = cv2.resize(gray_img, (128, 128))
cv2.imwrite("gray_images/gray_" +str(count) +".jpg", gray_img)
cv2.imwrite("color_images/color_" +str(count) +".jpg", image)
count += 1
#%%
path_test = r"D:\\nền\\bia.jpeg"
image = cv2.imread(path_test, 0)
cv2.imshow('image', image)
print(image.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
giay = list(data_dir.glob('giay/*'))
import os
import PIL
import PIL.Image
import numpy as np
np.set_printoptions(linewidth=200)
PIL.Image.open(str(giay[1]))
#%%

# batch_size = 32
# img_height = 180
# img_width = 180
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#%% load from local
import tensorflow as tf
import pathlib
# path = "D:\hoangnhi2310.github.io\hoangnhi2310.github.io\giaydep2.tar"
path = "D:\Hahalolo\persona\test_images\Giày dep\giaydep.tar"
data_dir = tf.keras.utils.get_file(origin=path, 
                                   fname='giaydep', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
image_count = len(list(data_dir.glob('*/*')))
print(image_count)
#%%
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
for f in list_ds.take(5):
  print(f.numpy())
#%%
giay = list(data_dir.glob('giay/*'))
import os
import PIL
import PIL.Image
import numpy as np
np.set_printoptions(linewidth=200)
PIL.Image.open(str(giay[1]))
