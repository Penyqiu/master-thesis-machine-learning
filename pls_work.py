import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Ścieżka do katalogu z danymi treningowymi
Datadirectory = '/home/kacper-penczynski/Pulpit/magisterka/master-thesis-machine-learning/resized'

Classes = ['awake', 'drowsy', 'low vigilant']

# Wymiary obrazu
img_size = 224

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = datagen.flow_from_directory(
    Datadirectory,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='binary',
    subset='training')

validation_generator = datagen.flow_from_directory(
    Datadirectory,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='binary',
    subset='validation')

num_train_samples = train_generator.samples
num_val_samples = validation_generator.samples
num_total_samples = num_train_samples + num_val_samples
num_classes = len(train_generator.class_indices)

print("Klasy: ", train_generator.class_indices)
print("Liczba klas: ", num_classes)
print("Liczba danych treningowych: ", num_train_samples)
print("Liczba danych walidacyjnych: ", num_val_samples)
print("Liczba ogólna wszystkich danych: ", num_total_samples)

base_model = tf.keras.applications.MobileNet(input_shape=(img_size, img_size, 3),
                                              include_top=False,
                                              weights='imagenet')
base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

new_model = tf.keras.Model(inputs=base_model.input, outputs=x)

new_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

history = new_model.fit(
    train_generator,
    steps_per_epoch=num_train_samples // 32,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=num_val_samples // 32,
    callbacks=[early_stopping, reduce_lr])

new_model.save('mobilenet_full_face.h5')
