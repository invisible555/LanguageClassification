import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
import concurrent.futures
import matplotlib.pyplot as plt
import skimage.io
import pickle
from imageprocessing import load_image
from audioprocessing import audio_to_spectogram_image


def load_history(filename):
    with open(filename, "rb") as file:
        history = pickle.load(file)
    return history

def save_history(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(history.history, file)
'''
def save_spectogram_to_png(audio_path_files,csv_path,output_files_path,label,max_files):
    image_output=output_files_path

    df = pd.read_csv(csv_path,sep='\t')
    image_paths = df[df['gender'] == label]['path'].tolist()
    image_paths=male_image_paths[:max_files]
    audio_to_spectogram_image.save_audio_to_png_gray_scale_from_directory(audio_path_files,file_output,image_paths)
'''

def create_model(input_shape,output_number):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape))) 
    model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(output_number, activation='softmax'))
    return model

def train_model(model, train_data, train_labels, test_data, test_labels, epochs=10):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data, test_labels))
    return history

def enocde_labels(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels) 
    labels = label_encoder.transform(labels)
    return labels

model = create_model((216,128,1),2)

data_female,labels_female = load_image.load_images_from_directory(0, 20000, "E:/cnn_network/female_gray", "female")
data_male,labels_male = load_image.load_images_from_directory(0, 20000, "E:/cnn_network/male_gray", "male")


data = np.concatenate((data_female, data_male), axis=0)
labels = np.concatenate((labels_female, labels_male), axis=0)

labels = enocde_labels(labels)  # kobieta = 0 mezczyzna  = 1
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=10, validation_data=(test_data,test_labels))
model.save('model_cnn_projekt_better.h5')
save_history(history,"history_better.pkl")

