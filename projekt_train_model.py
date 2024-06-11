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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import ReduceLROnPlateau

def load_history(filename):
    with open(filename, "rb") as file:
        history = pickle.load(file)
    return history

def save_history(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(history.history, file)

def save_ss(model,filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

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

def encode_labels(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels) 
    labels = label_encoder.transform(labels)
    return labels

def plot_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
model = create_model((216,128,1),2)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
#model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
conf_matrix = [[0,0],[0,0]]
max = 170000
splits = 5
add = int(max/splits)
start=0
end = start+add
history = ""

loss_list = []
accuracy_list = []
for i in range(splits):



    data_pl,labels_pl = load_image.load_images_from_directory(start, end, "E:/cnn_network/png_files_gray/pl_png", "pl",216,128)
    data_en,labels_en = load_image.load_images_from_directory(start, end, "E:/cnn_network/png_files_gray/en_png", "en",216,128)
    print(data_pl.shape)
    
    
    data = np.concatenate((data_en, data_pl), axis=0)
    labels = np.concatenate((labels_en, labels_pl), axis=0)
    data = np.expand_dims(data, axis=-1)
    data_pl=0
    data_en=0
    labels_pl=0
    labels_en=0
 

    labels = encode_labels(labels)  # kobieta = 0 mezczyzna  = 1 , polski = 1 angielski=0
    #train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(end)
    #wyłączyć validation data, przy ostatnim powtórzeniu wczytać jedynie dane do walidacji
    if end <= (max - (max * 0.2)):
        history = model.fit(data, labels, epochs=18,batch_size=20,callbacks=[reduce_lr])
    else:
        loss, accuracy = model.evaluate(data, labels, verbose=2)

        predictions = model.predict(data)
        predictions=np.argmax(predictions, axis=1)
        conf_matrix += confusion_matrix(labels,predictions )
        
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print(conf_matrix)
        
    model.save('model_cnn_language_not_resized3.h5')
    #save_history(history,"history_language4.pkl")
    save_ss(conf_matrix,"conf_matrix_not_resized3.pkl")


    start+= add
    end+=add


disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=["en","pl"])
disp.plot()
plt.show()
#plot_history(history)
print(np.array(accuracy_list).mean())
print(np.array(loss_list).mean())