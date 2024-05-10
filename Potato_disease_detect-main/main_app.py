

import numpy as np 
import streamlit as st 
import cv2 
from keras.models import load_model 
from tensorflow.keras import models  
import tensorflow as tf 
from tensorflow import keras

model = keras.models.load_model('final.h5')

class_names = [ 'Potato___Early_blight' , 'Potato___Late_blight' , 'Potato___healthy']

st.title('AGRI SCAN') 
st.title("Plant leaf disease detection")
st.markdown("Upload an image of the plant leaf")

plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict Disease')

if submit:
    if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Display the image.
        st.image(opencv_image, channels='BGR')
        st.write(opencv_image.shape)

        # Resizing the image.
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Convert the image to 4D.
        opencv_image = np.expand_dims(opencv_image, axis=0)

        # Make a prediction.
        y_pred = model.predict(opencv_image)
        result = class_names[np.argmax(y_pred)]
        st.title("This is a " + result.split("___")[1] + " leaf of " + result.split("___")[0])
