import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('/home/kacper-penczynski/Pulpit/magisterka/master-thesis-machine-learning/mobilenet_full_face.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model_from_video.tflite', 'wb') as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="model_from_video.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Pobierz wynik
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)