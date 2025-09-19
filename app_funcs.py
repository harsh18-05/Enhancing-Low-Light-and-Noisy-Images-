import streamlit as st
import numpy as np
import pandas as pd 
import os
import cv2 as cv
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from keras import backend as K
from keras.layers import add, Conv2D, Input
from keras.models import Model
# np.random.seed(1)
import warnings
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
warnings.simplefilter(action='ignore', category=FutureWarning)


@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def create_model(shape=(256, 256, 3)):
  in_ = Input(shape)
  p_in_ = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(in_)

  model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(p_in_)
  model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
  model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
  model_2 = Conv2D(32,(1,1), activation='relu',padding='same',strides=1)(p_in_)
  model_add = add([model_1,model_2])
  model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add)
  model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
  model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
  model_add_2 = add([model_add,model_3])

  model_1 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
  model_1 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_1)
  model_1 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_1)
  model_2 = Conv2D(64,(1,1), activation='relu',padding='same',strides=1)(model_add_2)
  model_add = add([model_1,model_2])
  model_3 = Conv2D(128,(3,3), activation='relu',padding='same',strides=1)(model_add)
  model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_3)
  model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_3)
  model_add_2 = add([model_add,model_3])

  model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
  model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
  model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
  model_2 = Conv2D(32,(1,1), activation='relu',padding='same',strides=1)(model_add_2)
  model_add = add([model_1,model_2])
  model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add)
  model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
  model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
  model_add_2 = add([model_add,model_3])
  out_ = Conv2D(3,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
  Model_Enhancer = Model(inputs=in_, outputs=out_)
  return Model_Enhancer


@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def instantiate_model(shape=(256, 256, 3)):
    model = create_model(shape)
    model.load_weights("Weights/")
    return model


@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def enhance_image(uploaded_image, downloaded_image):
    low_light_img = Image.open(uploaded_image).convert('RGB')
    width, height = low_light_img.size
    #low_light_img = low_light_img.resize((256,256),Image.NEAREST)
    model = instantiate_model(shape = (height, width, 3))

    image = np.array(low_light_img, dtype = np.float64)
    image = image / 255.0
    image = np.expand_dims(image, axis = 0)

    output = model.predict(image)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0,255)

    output_image = output_image.reshape((np.shape(output_image)[0],np.shape(output_image)[1],3))
    output_image = np.uint32(output_image)
    final_image = Image.fromarray(output_image.astype('uint8'),'RGB')
    final_image.save(downloaded_image)


@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def download_success():
    st.balloons()
    st.success('✅ Downloaded Successfully !!')
