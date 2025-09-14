import cv2 
import numpy as np 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array 
 
# Load trained model 
model = load_model('traffic_sign_model.h5') 
 
# Define label mapping 
classes = {0:'Speed Limit 20km/h', 1:'Speed Limit 30km/h', ..., 42:'End of no 
passing'} 
 
10 
 
def preprocess(img): 
    img = cv2.resize(img, (64, 64)) 
    img = img_to_array(img) 
    img = img / 255.0 
    img = np.expand_dims(img, axis=0) 
    return img 
 
cap = cv2.VideoCapture(0) 
 
while True: 
    ret, frame = cap.read() 
    if not ret: 
        break 
 
    roi = cv2.resize(frame, (64, 64)) 
    image = preprocess(roi) 
 
    prediction = model.predict(image) 
    class_id = np.argmax(prediction) 
    confidence = prediction[0][class_id] 
 
    if confidence > 0.8: 
        label = classes[class_id] 
        cv2.putText(frame, f'{label} ({confidence:.2f})', (10, 40),  
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) 
 
    cv2.imshow('Real-Time Traffic Sign Detection', frame) 
 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
 
cap.release() 
cv2.destroyAllWindows()
