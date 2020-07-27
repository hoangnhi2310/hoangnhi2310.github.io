# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:02:46 2020

@author: ASUS
"""
import tensorflow as tf
import pathlib
dataset_url = "https://hoangnhi2310.github.io/giaydep.tar"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='giaydep', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.JPG')))
print(image_count)
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
path = "D:\hoangnhi2310.github.io\hoangnhi2310.github.io"
data_dir = tf.keras.utils.get_file(origin=path, 
                                   fname='giaydep', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
image_count = len(list(data_dir.glob('*/*.JPG')))
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
