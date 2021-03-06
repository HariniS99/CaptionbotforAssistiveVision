# -*- coding: utf-8 -*-
"""LSTM_intermediate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k0S6FQqd0-VNz0q6BCbJamdF9QJ5ul3a
"""

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
#from nltk.tokenize import word_tokenize
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
from keras import layers
from keras.models import load_model

#from google.colab import drive
#drive.mount('/content/drive')



import pickle

with open("./output/tokenizer.pkl", "rb") as token:
    tokenizer = pickle.load(token)

modelvgg = VGG16(include_top=True, weights=None)
modelvgg.load_weights("./model/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
modelvgg.layers.pop()
modelvgg = Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-2].output)

with open('./output/features.pkl', 'rb') as f:
    images = pickle.load(f)






from keras import layers
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
vocab_size = 4476
maxlen = 30
## image feature
dim_embedding = 64

input_image = layers.Input(shape=(4096,))
fe1 = Dropout(0.5)(input_image)
fimage = layers.Dense(256,activation='relu',name="ImageFeature")(fe1)
## sequence model
input_txt = layers.Input(shape=(maxlen,))
ftxt = layers.Embedding(vocab_size,256, mask_zero=True)(input_txt)
ftxt = layers.LSTM(256,name="CaptionFeature",return_sequences=True)(ftxt)
#,return_sequences=True
#,activation='relu'
se2 = Dropout(0.5)(ftxt)
ftxt = layers.LSTM(256,name="CaptionFeature2")(se2)
## combined model for decoder
decoder = layers.add([ftxt,fimage])
decoder = layers.Dense(256,activation='relu')(decoder)
output = layers.Dense(vocab_size,activation='softmax')(decoder)
model = Model(inputs=[input_image, input_txt],outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam')

#print(model.summary())

for i in range(len(model.layers)):
    layer=model.layers[i]
    
    #print(i,layer.name ,layer.output.shape)

for i in range(len(model.layers)):
    layer=model.layers[i]
    if 'dropout' in layer.name:
        continue
    #print(i,layer.name ,layer.output.shape)



model = load_model('./model/new_model_epoch15.h5')


def get_img_seq(filename):
    maxlen=30
    npic = 5
    npix = 224
    target_size = (npix,npix,3)
    count = 1
    fig = plt.figure(figsize=(10,20))

    image = load_img(filename, target_size=target_size)
    image = img_to_array(image)  
    n_image = preprocess_input(image)
    Self_made_img = modelvgg.predict(n_image.reshape( (1,) + n_image.shape[:3])) 
    image_ = Self_made_img.flatten();

    image=image_.reshape(1,len(image_))
    in_text = 'startseq'
    for iword in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence],maxlen)
    return image,sequence

#filename='/content/drive/MyDrive/finalyearproject/test_img/Img_4.jpg'
#image,sequence=get_img_seq(filename)

#vgg_lstm1 = models.Model(inputs=model.input,outputs=model.layers[1].output)
#layer1=vgg_lstm1([image,sequence])
#vgg_lstm2 = models.Model(inputs=model.input,outputs=model.layers[2].output)
#layer2=vgg_lstm2([image,sequence])
#vgg_lstm3 = models.Model(inputs=model.input,outputs=model.layers[3].output)
#layer3=vgg_lstm3([image,sequence])
#vgg_lstm6 = models.Model(inputs=model.input,outputs=model.layers[6].output)
#layer6=vgg_lstm6([image,sequence])
#vgg_lstm7 = models.Model(inputs=model.input,outputs=model.layers[7].output)
#layer7=vgg_lstm7([image,sequence])
#vgg_lstm8 = models.Model(inputs=model.input,outputs=model.layers[8].output)
#layer8=vgg_lstm8([image,sequence])
#vgg_lstm9 = models.Model(inputs=model.input,outputs=model.layers[9].output)
#layer9=vgg_lstm9([image,sequence])
#vgg_lstm10 = models.Model(inputs=model.input,outputs=model.layers[10].output)
#layer10=vgg_lstm10([image,sequence])





def img_seq_model1(filename):
    image, sequence = get_img_seq(filename)

    vgg_lstm1 = Model(inputs=model.input, outputs=model.layers[1].output)
    layer1 = vgg_lstm1([image, sequence])
    # vgg_lstm10 = Model(inputs=model.input, outputs=model.layers[10].output)
    # layer10 = vgg_lstm10([image, sequence])

    return layer1

def img_seq_model10(filename):
    image, sequence = get_img_seq(filename)

    # vgg_lstm1 = Model(inputs=model.input, outputs=model.layers[1].output)
    # layer1 = vgg_lstm1([image, sequence])
    vgg_lstm10 = Model(inputs=model.input, outputs=model.layers[9].output)
    layer10 = vgg_lstm10([image, sequence])
    return layer10

