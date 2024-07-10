import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
from audioprocessing import audio_to_spectogram_image
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Rescaling

MODEL_PATH="cnn_network_lang"
INPUT_DIR = "E:/AI_models/gender_photos"
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 216 
BATCH_SIZE = 32
N_CHANNELS = 1
N_CLASSES = 2
EPOCHS=10



def create_model(input_shape,output_number):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape))) 
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(512, 3, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(256, 3,padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3))) 
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3))) 
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(output_number, activation='softmax'))
    return model

def train_model(model, train_dataset, valid_dataset,bath_szie, epochs=10):    
    log_dir = "logs/fit3/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',  # Metryka do monitorowania
        factor=0.1,  # Czynnik zmniejszający współczynnik uczenia
        patience=2,  # Liczba epok bez poprawy przed zmniejszeniem współczynnika uczenia
        min_lr=0.00001
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset,batch_size=bath_szie, epochs=epochs, validation_data=valid_dataset,callbacks=[tensorboard_callback,reduce_lr_callback])
    return model

def model_save(model,path):
    model.save(path)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=INPUT_DIR,
                                             shuffle=True,
                                             color_mode='grayscale',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="training",
                                             seed=0)
                                            
                      
                                 
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=INPUT_DIR,
                                             shuffle=True,
                                             color_mode='grayscale',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="validation",
                                             seed=0)

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
valid_dataset = valid_dataset.map(lambda x, y: (normalization_layer(x), y))


model = create_model((IMAGE_HEIGHT,IMAGE_WIDTH,N_CHANNELS),N_CLASSES)

model=train_model(model,train_dataset,valid_dataset,BATCH_SIZE,EPOCHS)

model_save(model,MODEL_PATH)