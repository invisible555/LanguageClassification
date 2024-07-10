import librosa
import skimage.io
import os
import numpy as np

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def create_spectrogram(y,sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.min)
    return S_DB


def save_segment_to_directory_gray_scale(folder_name, folder_output, segment_length=5):
    licznik = 0
    for filename in os.listdir(folder_name):
        filepath_full = os.path.join(folder_name, filename)
        if os.path.isfile(filepath_full):
            y, sr = librosa.load(filepath_full, sr=22050)
            segment_length_samples = segment_length * sr
            total_length_samples = len(y)
            if total_length_samples < segment_length_samples:
                pass
                # print(f"Plik {filename} jest za krótki i został pominięty.")
            else:
                segments = []
                for i in range(0, total_length_samples, sr):  # Przesunięcie o 1 sekundę (sr próbek)
                    if i + segment_length_samples <= total_length_samples:
                        segment = y[i:i + segment_length_samples]
                        segments.append(segment)
                    else:
                        break  # Zapewnia, że nie wyjdziemy poza zakres
                for segment in segments:
                    spectrogram = create_spectrogram(segment, sr)
                    img = scale_minmax(spectrogram, 0, 255).astype(np.uint8)
                    img = 255 - img
                    filename_without_extension = os.path.splitext(filename)[0]
                    output_filepath = os.path.join(folder_output, f"{filename_without_extension}_{licznik}.png")
                    skimage.io.imsave(output_filepath, img)
                    licznik += 1




if __name__ == "__main__":
    pass
