import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
import pickle
import matplotlib.pyplot as plt
from imageprocessing import load_image
from audioprocessing import audio_to_spectogram_image
from audioprocessing import audio_to_spectogram
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

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

def zbierz_zakresy_jezykow(tablica):
    zakresy_jezykow = []
    poprzedni_jezyk = None
    zakres_poczatek = 0

    for i in range(len(tablica)):
        jezyk = label_map[tablica[i]]
        if jezyk != poprzedni_jezyk:
            if poprzedni_jezyk is not None:
                zakresy_jezykow.append((zakres_poczatek, i - 1, poprzedni_jezyk))
            poprzedni_jezyk = jezyk
            zakres_poczatek = i

    # Sprawdzenie dla ostatniej cyfry
    ostatnia_cyfra = label_map[tablica[-1]]
    if ostatnia_cyfra != poprzedni_jezyk:
        zakresy_jezykow.append((zakres_poczatek, len(tablica) - 1, ostatnia_cyfra))

    # Sprawdzenie, czy są brakujące sekundy
    if len(zakresy_jezykow) > 0:
        ostatni_koniec = zakresy_jezykow[-1][1]
        ostatni_jezyk = zakresy_jezykow[-1][2]
        if ostatni_koniec < len(tablica) -1 :
            zakres_poczatek = ostatni_koniec + 1
            ostatni_koniec += 6
            zakresy_jezykow.append((zakres_poczatek, ostatni_koniec, ostatni_jezyk))

    return zakresy_jezykow


model_path = 'model_cnn_language_not_resized2.h5'
history_path = "history_cnn_projekt"
folder_path= "plikidzwiekowe/"
label_map = {0: 'en', 1: 'pl'}
'''
data_female,labels_female = load_image.load_images_from_directory(0, 20000, "E:/cnn_network/female_gray", "female",216,128)
data_male,labels_male = load_image.load_images_from_directory(0, 20000, "E:/cnn_network/male_gray", "male",216,128)
data = np.concatenate((data_female, data_male), axis=0)
labels = np.concatenate((labels_female, labels_male), axis=0)
labels = enocde_labels(labels)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
'''
#prediction_data = audio_to_spectogram_image.audio_to_png_gray_scale(folder_path)

prediction_data,_ = audio_to_spectogram.load_and_process_audio("pl_internet2.mp3",folder_path,"pl")
print(prediction_data.shape)

#np.set_printoptions(threshold=np.inf, linewidth=np.inf)
start_time=0
end_time=0
#print(prediction_data)
#test_model(test_data, test_labels, model_path)
#show_train_history(history_path)
if(len(prediction_data)):
    start_time = time.time()
    predictions = make_prediction(prediction_data,model_path)
    end_time = time.time()
    print(predictions)
    predicted_indexes = np.argmax(predictions, axis=1)
    print(predicted_indexes)
    unique, counts = np.unique(predicted_indexes, return_counts=True)
    most_common_index = np.argmax(counts)
    most_common_value = unique[most_common_index]
    most_common_label = label_map[most_common_value]
    print("Najczęściej występująca etykieta:", most_common_label)
    zakresy = zbierz_zakresy_jezykow(predicted_indexes)
    print(zakresy)


else:
    print("Za krótki plik")

unikalne, liczba_wystapien = np.unique(predicted_indexes, return_counts=True)
for wartosc, wystapienia in zip(unikalne, liczba_wystapien):
    print(f"Liczba {label_map[wartosc]} występuje {wystapienia} razy.")
execution_time = end_time - start_time
print("Czas wykonania:", execution_time, "sekund")