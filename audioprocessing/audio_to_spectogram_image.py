import librosa
import matplotlib.pyplot as plt
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

def save_audio_to_png_gray_scale_from_directory(folder_name, folder_output, filenames):
    for filename in filenames:  
        filepath_full = os.path.join(folder_name, filename)
        y, sr = librosa.load(filepath_full)
        S_DB = create_spectrogram(y,sr)
        S_DB = 255 - S_DB
        img = scale_minmax(S_DB, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) 
        output_filepath = os.path.join(folder_output, filename + ".png")
        skimage.io.imsave(output_filepath, img)




def save_audio_to_png_from_directory(folder_name, folder_output, filenames):
    for filename in filenames:  
        filepath_full = os.path.join(folder_name, filename)
        y, sr = librosa.load(filepath_full)
        S_DB = create_spectrogram(y,sr)
        librosa.display.specshow(S_DB, sr=sr)
        plt.savefig(os.path.join(folder_output, os.path.splitext(filename)[0] + ".png"))
        plt.close()


'''
def audio_to_png_gray_scale(audio_filepath):
    y, sr = librosa.load(audio_filepath)
    S_DB = create_spectrogram(y, sr)
    img = 255 - img  
    img = scale_minmax(S_DB, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  
    img_resized = skimage.transform.resize(img, (216, 128), anti_aliasing=True)
    img = np.expand_dims(img_resized, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

'''
