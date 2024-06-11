import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

import skimage.io

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def load_and_process_audio(filename, folderpath, label, segment_length=5):
    data = []
    labels = []
    desired_height=216
    desired_width=128
    filepath_full = os.path.join(folderpath, filename)
    y, sr = librosa.load(filepath_full,sr=22050)

    # Obliczenie długości segmentu w próbkach
    segment_length_samples = segment_length * sr
    total_length_samples = len(y)
    
    # Jeśli długość audio jest mniejsza niż segment_length, zignoruj plik
    if total_length_samples < segment_length_samples:
        print(f"Plik {filename} jest za krótki i został pominięty.")
        return np.array(data), labels
    
    # Podział audio na segmenty 5-sekundowe z przesunięciem o 1 sekundę
    segments = []
    for i in range(0, total_length_samples, sr):  # Przesunięcie o 1 sekundę (sr próbek)
        if i + segment_length_samples <= total_length_samples:
            segment = y[i:i + segment_length_samples]
            segments.append(segment)
        else:
            break  # Zapewnia, że nie wyjdziemy poza zakres
    
    for segment in segments:
        spectogram = create_spectogram(segment, sr)
        
        img = scale_minmax(spectogram, 0, 255).astype(np.uint8)
        img = 255 - img
        img = np.flip(img, axis=0) 
        img = np.expand_dims(img, axis=-1)
        img = skimage.img_as_float32(img)
        #img = skimage.transform.resize(img, (desired_height, desired_width))
        data.append(img)
        labels.append(label)
    
    data = np.array(data)
  
    print(f"PL data z pliku {filename}: ", len(data))
    return data, labels

def create_spectogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.min)
    return S_DB

if __name__ == "__main__":
    pass





