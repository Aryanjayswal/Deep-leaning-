
import cv2
import numpy as np
import tensorflow as tf 
from tensorflow import keras 
from keras  import models
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import img_to_array


# ------------------------------
model = load_model("F:\Mask Detection Project Opencv\mask_detector_model.h5") # Change path as needed

# ------------------------------
# Load Haar Cascade Face Detector
# ------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------------------
# Prediction Buffer for Smoothing
# ------------------------------
pred_buffer = deque(maxlen=5)

# ------------------------------
# Function: Resize with Aspect Ratio & Padding
# ------------------------------
def resize_with_padding(image, target_size=(96, 96)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_w = target_size[1] - new_w
    pad_h = target_size[0] - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded

# ------------------------------
# Open Webcam
# ------------------------------
cap = cv2.VideoCapture(0)
window_name = "Enhanced Mask Detection"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        if face.size > 0:
            # Lighting normalization
            face_yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
            face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
            face = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)

            # Convert to RGB (if model trained in RGB)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Resize with aspect ratio & padding
            face_resized = resize_with_padding(face, (96, 96))
            face_resized = face_resized.astype("float") / 255.0
            face_resized = img_to_array(face_resized)
            face_resized = np.expand_dims(face_resized, axis=0)

            # Predict
            pred = model.predict(face_resized, verbose=0)[0][0]
            pred_buffer.append(pred)
            avg_pred = np.mean(pred_buffer)

            # Decision based on smoothed prediction
            if avg_pred < 0.5:
                label = "With Mask"
                conf = 1 - avg_pred
                color = (0, 255, 0)
            else:
                label = "Without Mask"
                conf = avg_pred
                color = (0, 0, 255)

            # Draw results
            label_text = f"{label}: {conf:.2f}"
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()