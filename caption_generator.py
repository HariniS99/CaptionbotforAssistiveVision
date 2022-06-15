#!/usr/bin/env python
# coding: utf-8

# In[1]:




import keras
import tensorflow as tf

import numpy as np




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


# In[4]:


modelvgg = VGG16(include_top=True,weights=None)
modelvgg.load_weights("./model/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
modelvgg.layers.pop()
modelvgg = Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-2].output)


# In[5]:


import pickle
from keras.preprocessing.text import Tokenizer


# In[6]:


with open("./output/tokenizer.pkl", "rb") as token:
    tokenizer = pickle.load(token)


# In[7]:


index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])


# In[11]:


model=load_model('./model/model.h5')


# In[22]:


def test_images(filename):
    maxlen=30
       
    def predict_caption(image):
  
        in_text = 'startseq'
        for iword in range(maxlen):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence],maxlen)
            yhat = model.predict([image,sequence],verbose=0)
            yhat = np.argmax(yhat)
            newword = index_word[yhat]
            in_text += " " + newword
            if newword == "endseq":
                break
        return(in_text)

    npic = 5
    npix = 224
    target_size = (npix,npix,3)
    count = 1
    #fig = plt.figure(figsize=(10,20))

    # Images 
    image = load_img(filename, target_size=target_size)
    image = img_to_array(image)  # Convert the image pixels to a Numpy Array
    n_image = preprocess_input(image)
    Self_made_img = modelvgg.predict(n_image.reshape( (1,) + n_image.shape[:3])) 
    image_ = Self_made_img.flatten();
    
    #image_load = load_img(filename, target_size=target_size)
    #ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])
    #ax.imshow(image_load)
    #count += 1

    # Captions
    caption = predict_caption(image_.reshape(1,len(image_)))
    y_cap=caption.split()
    y_cap = y_cap[1:-1]
    new_caption=' '.join(y_cap)
    #print(y_cap[1:-1])
    #txt_to_speech(caption,filename)

    #ax = fig.add_subplot(npic,2,count)
    #plt.axis('off')
    #ax.plot()
    #ax.set_xlim(0,1)
    #ax.set_ylim(0,1)
    #ax.text(0,0.5,caption,fontsize=15)
    #count += 1

    #plt.show()
    return new_caption


# In[34]:


#result=test_images('C:/Users/DELL/Documents/Projects/flickr/test_img/Img_9.jpg')


# In[35]:


#result


# In[ ]:




