import cv2
import os
from keras.models import load_model
import numpy as np

# Load the model
model_path = '/home/kacper-penczynski/Pulpit/magisterka/master-thesis-machine-learning/mobilenet.h5'
model = load_model(model_path)

# Initialize face and eyes detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)
print("Video stream open." if cap.isOpened() else "Problem opening video stream.")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Starting values for closed-eyes inference
score = 0
threshold = 6
thicc = 2
rpred = [99]
lpred = [99]

# Infinite loop for frame capture, inference, and scoring
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    height, width = frame.shape[:2]
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces and eyes
    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = left_eye_cascade.detectMultiScale(gray)
    right_eye = right_eye_cascade.detectMultiScale(gray)
    
    # Draw black bars at the top and bottom
    cv2.rectangle(frame, (0, height-50), (width, height), (0,0,0), thickness=cv2.FILLED)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    # Process and predict for each eye
    for eye, eye_cascade in [('right', right_eye), ('left', left_eye)]:
        for (x, y, w, h) in eye_cascade:
            eye_frame = frame[y:y+h, x:x+w]
            eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
            eye_frame = cv2.resize(eye_frame, (224, 224))
            eye_frame = eye_frame / 255
            eye_frame = np.repeat(eye_frame[..., np.newaxis], 3, axis=2)  # Convert grayscale to 3 channels
            eye_frame = np.expand_dims(eye_frame, axis=0)
            prediction = model.predict(eye_frame)[0][0]
            status = "open" if prediction > 0.5 else "closed"
            print(f"Model confidence for {eye} eye: {prediction:.2f} ({status})")
            if eye == 'right':
                rpred = [prediction]
            else:
                lpred = [prediction]
            break

    # Determine if both eyes are closed and update the score accordingly
    if rpred[0] < 0.8 and lpred[0] < 0.8:
        score += 1
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        if score > threshold + 1:
            score = threshold  # Prevent runaway score
    else:
        score = max(score - 1, 0)
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the current drowsiness score
    cv2.putText(frame, f'Drowsiness Score:{score}', (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Indicate excessive drowsiness with a red border
    if score > threshold:
        thicc = min(max(thicc + 2, 2), 16) if thicc < 16 else max(thicc - 2, 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thickness=thicc)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()