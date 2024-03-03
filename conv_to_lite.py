# Importowanie niezbędnych bibliotek
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Załaduj model H5
model = load_model('/home/kacper-penczynski/Pulpit/magisterka/master-thesis-machine-learning/mobilenet.h5')

# Konwersja modelu na format TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Zapisz model TensorFlow Lite do pliku
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Testowanie przekonwertowanego modelu TensorFlow Lite
# Załaduj model TFLite i przygotuj interpreter
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Pobierz informacje o wejściu i wyjściu
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Testuj model (przykład danych wejściowych)

# Przygotuj dane wejściowe (upewnij się, że są one w odpowiednim formacie i rozmiarze)
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Pobierz wynik
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)