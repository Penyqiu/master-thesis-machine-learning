import cv2
import numpy as np
from keras.models import load_model
import time

model_path = '/home/kacper-penczynski/Pulpit/magisterka/master-thesis-machine-learning/mobilenet.h5'
model = load_model(model_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Problem podczas otwierania strumienia wideo.")
    exit()

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

score = 0
threshold = 6
thicc = 1
measurements = []
collect_measurements = False

print("Naciśnij Enter, aby zacząć zbierać pomiary.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się przechwycić klatki.")
        break

    if cv2.waitKey(1) == 13:  # 13 to kod ASCII dla Enter
        collect_measurements = True
        print("Rozpoczęto zbieranie pomiarów.")

    if collect_measurements:
        height, width = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
        left_eye = left_eye_cascade.detectMultiScale(gray)
        right_eye = right_eye_cascade.detectMultiScale(gray)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

        for eye, eye_cascade in [('right', right_eye), ('left', left_eye)]:
            for (x, y, w, h) in eye_cascade:
                eye_frame = frame[y:y+h, x:x+w]
                eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
                eye_frame = cv2.resize(eye_frame, (224, 224))
                eye_frame = eye_frame / 255
                eye_frame = np.repeat(eye_frame[..., np.newaxis], 3, axis=2) 
                eye_frame = np.expand_dims(eye_frame, axis=0)
                prediction = model.predict(eye_frame)[0][0]
                status = "zamknięte" if prediction > 0.9 else "otwarte"
                measurements.append((prediction if eye == 'left' else 0, prediction if eye == 'right' else 0))
                break

        if len(measurements) >= 1000:  # Zakończenie zbierania po 1000 pomiarach
            break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Obliczenie średnich wartości przewidywań dla lewego i prawego oka
average_lpred = sum([m[0] for m in measurements]) / len(measurements)
average_rpred = sum([m[1] for m in measurements]) / len(measurements)

print(f"Średnia pewność dla lewego oka: {average_lpred:.2f}")
print(f"Średnia pewność dla prawego oka: {average_rpred:.2f}")
