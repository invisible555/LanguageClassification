import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from audioprocessing import audio_to_spectogram_image
from audioprocessing import audio_to_spectogram
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def test_model(test_data, test_labels, model_path):
    loaded_model = tf.keras.models.load_model(model_path)
    loss, accuracy = loaded_model.evaluate(test_data, test_labels, verbose=2)
    print(f"Loss: {loss}, accuracy: {accuracy}")


def make_prediction(prediction_data, model_path): # probability for each label
    loaded_model = tf.keras.models.load_model(model_path)
    predictions = loaded_model.predict(prediction_data)
    return predictions

def show_predicted_label(prediction_data, model_path): # predicted label
    predictions = make_prediction(prediction_data, model_path)
    predicted_indexes = np.argmax(predictions, axis=1)
    return predicted_indexes

def most_common_label(predicted_indexes):
    unique, counts = np.unique(predicted_indexes, return_counts=True)
    most_common_index = np.argmax(counts)
    most_common_value = unique[most_common_index]
    most_common_label = label_map[most_common_value]
    return most_common_label

def enocde_labels(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)  # kobieta =0 mezczyzna  = 1
    labels = label_encoder.transform(labels)
    return labels

def count_prediction_for_each_label(predicted_indexes):
    unikalne, liczba_wystapien = np.unique(predicted_indexes, return_counts=True)
    return unikalne,liczba_wystapien




model_path = 'model_cnn_language7.h5'
folder_path= "plikidzwiekowe/"
label_map = {0: 'en', 1: 'pl'}



prediction_data,_ = audio_to_spectogram.load_and_process_audio("C:/Users/niewi/Desktop/kwalifikacjajezyka/plikidzwiekowe/ang1.mp3",folder_path,"pl")

if(len(prediction_data)):
    predictions = make_prediction(prediction_data,model_path)
    print(predictions)
    predicted_indexes = show_predicted_label(prediction_data,model_path)
    print(predicted_indexes)
    predicted_label = most_common_label(predicted_indexes)
    print("Najczęściej występująca etykieta:", predicted_label)
    unikalne, liczba_wystapien = count_prediction_for_each_label(predicted_indexes)
    for wartosc, wystapienia in zip(unikalne, liczba_wystapien):
        print(f"Liczba {label_map[wartosc]} występuje {wystapienia} razy.")

else:
    print("Za krótki plik")


