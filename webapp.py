# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 22:16:51 2022

@author: Akash
"""

import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from resizeimage import resizeimage


MODEL = tf.keras.models.load_model("C:/Users\91866\Desktop\Great_Learning\DSBA\Weld_Quality_Detection\SavedModel")

CLASS_NAMES = ["bad_quality_weld", "good_quality_weld"]



def read_file_as_image(input_data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(input_data)))
    return image

def predict(
    input_data
):
    image = read_file_as_image(input_data.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

def main():
    st.title('Weld Quality Detection')
    
    Upload_Image = (st.file_uploader('Take a picture of weld defect'))
    
    detection_of_weld_defect = ''
    
    if st.button('Detect'):
        detection_of_weld_defect = predict(Upload_Image)
        show_image = st.image(Upload_Image)
        
    st.success(detection_of_weld_defect)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    