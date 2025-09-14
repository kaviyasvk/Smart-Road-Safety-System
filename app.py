app.py 
import streamlit as st 
import cv2 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array 
import numpy as np 
 
# Load model 
model = load_model('traffic_sign_model.h5') 
classes = {0:'Speed Limit 20km/h', 1:'Speed Limit 30km/h', ..., 42:'End of no 
passing'} 
 
def preprocess(image): 
    image = cv2.resize(image, (64, 64)) 
    image = img_to_array(image) 
    image = image / 255.0 
    image = np.expand_dims(image, axis=0) 
    return image 
 
st.title("           Real-Time Traffic Sign Detection") 
 
run = st.checkbox('Run Webcam') 
FRAME_WINDOW = st.image([]) 
 
cap = cv2.VideoCapture(0) 
 
while run: 
    ret, frame = cap.read() 
    if not ret: 
        break 
 
    image = preprocess(frame) 
    prediction = model.predict(image) 
    class_id = np.argmax(prediction) 
    confidence = prediction[0][class_id] 
 
    if confidence > 0.8: 
        label = f"{classes[class_id]} ({confidence:.2f})" 
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 
(0,255,0), 2) 
 
 
12 
 
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
cap.release() 
