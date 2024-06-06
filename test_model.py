import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
import pickle
import matplotlib.pyplot as plt
from imageprocessing import load_image
from audioprocessing import audio_to_spectogram_image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_history(filename):
    with open(filename, "rb") as file:
        history = pickle.load(file)
    return history

def save_history(history, filename):
    with open(filename, "wb") as file:
        pickle.dump(history, file)

def plot_history(history):
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def test_model(test_data, test_labels, model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    loss, accuracy = loaded_model.evaluate(test_data, test_labels, verbose=2)
    print(f"Loss: {loss}, accuracy: {accuracy}")

def show_train_history(history_path):
    loaded_history = load_history(history_path)
    plot_history(loaded_history)

def make_prediction(prediction_data, model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    predictions = loaded_model.predict(prediction_data)
    return predictions

def enocde_labels(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)  # kobieta =0 mezczyzna  = 1
    labels = label_encoder.transform(labels)
    return labels

model_path = 'model_cnn_projekt_better.h5'
history_path = "history_cnn_projekt"

data_female,labels_female = load_image.load_images_from_directory(0, 20000, "E:/cnn_network/female_gray", "female",216,128)
data_male,labels_male = load_image.load_images_from_directory(0, 20000, "E:/cnn_network/male_gray", "male",216,128)
data = np.concatenate((data_female, data_male), axis=0)
labels = np.concatenate((labels_female, labels_male), axis=0)
labels = enocde_labels(labels)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

prediction_data = audio_to_spectogram_image.audio_to_png_gray_scale("C:/Users/niewi/Desktop/langclassification/plikidzwiekowe/zmiksowane2.mp3")

test_model(test_data, test_labels, model_path)
show_train_history(history_path)
print(make_prediction(prediction_data,model_path))