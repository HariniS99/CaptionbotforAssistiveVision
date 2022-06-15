#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import sys, time, os, warnings
import numpy as np
import pandas as pd
from collections import Counter 
from keras.preprocessing.image import load_img



# In[2]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.utils import to_categorical
from keras import layers
from keras.models import load_model


# In[3]:


modelvgg = VGG16(include_top=True,weights=None)
modelvgg.load_weights("./model/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
modelvgg.layers.pop()
modelvgg = Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-2].output)


# In[5]:


#modelvgg.summary()


# In[6]:


import pickle 
from keras.preprocessing.text import Tokenizer


# In[7]:


with open("./output/tokenizer.pkl", "rb") as token:
    tokenizer = pickle.load(token)


# In[8]:


index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])


# In[9]:


model=load_model('./model/model.h5')


# In[10]:


from numpy import expand_dims
import matplotlib.pyplot as plt
from PIL import Image


# In[22]:


def f_blk2(filename):
    #ixs = [2, 5, 9, 13, 17]
    outputs = [modelvgg.layers[2].output]
    model3 =Model(inputs=modelvgg.inputs, outputs=outputs)
    head, tail = os.path.split(filename)
    tail=tail[:-4]

    img = load_img(filename, target_size=(224, 224))

    img = img_to_array(img)

    img = expand_dims(img, axis=0)

    img = preprocess_input(img)

    feature_maps = model3.predict(img)

    square = 8
    ix = 1
        
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='RdYlBu_r')
            ix += 1
            fn=tail+'_'+'layer2'
                
    plt.savefig('./static/fmap/'+fn+'.jpg')

    #plt.show()


# In[23]:


def f_blk5(filename):
    #ixs = [2, 5, 9, 13, 17]
    outputs = [modelvgg.layers[5].output]
    model3 =Model(inputs=modelvgg.inputs, outputs=outputs)
    head, tail = os.path.split(filename)
    tail=tail[:-4]

    img = load_img(filename, target_size=(224, 224))

    img = img_to_array(img)

    img = expand_dims(img, axis=0)

    img = preprocess_input(img)

    feature_maps = model3.predict(img)

    square = 8
    ix = 1
        
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='RdYlBu_r')
            ix += 1
            fn=tail+'_'+'layer5'
                
    plt.savefig('./static/fmap/'+fn+'.jpg')

    #plt.show()


# In[24]:


def f_blk9(filename):
    #ixs = [2, 5, 9, 13, 17]
    outputs = [modelvgg.layers[9].output]
    model3 =Model(inputs=modelvgg.inputs, outputs=outputs)
    head, tail = os.path.split(filename)
    tail=tail[:-4]

    img = load_img(filename, target_size=(224, 224))

    img = img_to_array(img)

    img = expand_dims(img, axis=0)

    img = preprocess_input(img)

    feature_maps = model3.predict(img)

    square = 8
    ix = 1
        
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='RdYlBu_r')
            ix += 1
            fn=tail+'_'+'layer9'
                
    plt.savefig('./static/fmap/'+fn+'.jpg')

    #plt.show()


# In[25]:


def f_blk13(filename):
    #ixs = [2, 5, 9, 13, 17]
    outputs = [modelvgg.layers[13].output]
    model3 =Model(inputs=modelvgg.inputs, outputs=outputs)
    head, tail = os.path.split(filename)
    tail=tail[:-4]

    img = load_img(filename, target_size=(224, 224))

    img = img_to_array(img)

    img = expand_dims(img, axis=0)

    img = preprocess_input(img)

    feature_maps = model3.predict(img)

    square = 8
    ix = 1
        
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='RdYlBu_r')
            ix += 1
            fn=tail+'_'+'layer13'
                
    plt.savefig('./static/fmap/'+fn+'.jpg')

    #plt.show()


# In[26]:


def f_blk17(filename):
    #ixs = [2, 5, 9, 13, 17]
    outputs = [modelvgg.layers[17].output]
    model3 =Model(inputs=modelvgg.inputs, outputs=outputs)
    head, tail = os.path.split(filename)
    tail=tail[:-4]

    img = load_img(filename, target_size=(224, 224))

    img = img_to_array(img)

    img = expand_dims(img, axis=0)

    img = preprocess_input(img)

    feature_maps = model3.predict(img)

    square = 8
    ix = 1
        
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='RdYlBu_r')
            ix += 1
            fn=tail+'_'+'layer17'
                
    plt.savefig('./static/fmap/'+fn+'.jpg')

    #plt.show()


# In[27]:


#file='C:/Users/DELL/Documents/Projects/flickr/test_img/Img_8.jpg'
def get_fmap(file):
    f_blk2(file)
    f_blk5(file)
    f_blk9(file)
    f_blk13(file)
    f_blk17(file)


# In[ ]:




