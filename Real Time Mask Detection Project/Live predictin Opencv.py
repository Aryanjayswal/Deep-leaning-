
import cv2
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from keras  import models
from keras.models import load_model
from keras.preprocessing.image import img_to_array



model = load_model("F:/opencv/mask_detector_model.h5")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
window_name = "Live Mask Detection"

while True:
    ret, frame = cap.read()
    if not ret:
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        try:
           
            face_resized = cv2.resize(face, (96, 96))  
            face_resized = face_resized.astype("float") / 255.0
            face_resized = img_to_array(face_resized)
            face_resized = np.expand_dims(face_resized, axis=0)

           
            prediction = model.predict(face_resized)[0][0]

            if prediction < 0.5:
                label = "With Mask"
                confidence = 1 - prediction
                color = (0, 255, 0)
            else:
                label = "Without Mask"
                confidence = prediction
                color = (0, 0, 255)

         
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        except Exception as e:
            print("Error processing face:", e)

    
    cv2.imshow(window_name, frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()