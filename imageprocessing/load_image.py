import skimage.io
import tensorflow as tf
import os
import numpy as np
import pandas as pd


def load_images_from_directory(start, end, filepath, label,desired_height, desired_width,):
    data = []
    labels = []

    for filename in os.listdir(filepath)[start:end]:  
        if filename.endswith((".png")):
            filepath_full = os.path.join(filepath, filename)
            image = skimage.io.imread(filepath_full)
            image = skimage.img_as_float32(image)
     
            #image = skimage.transform.resize(image, (desired_height, desired_width))
            data.append(image)
            labels.append(label)  

    labels = np.array(labels)
    data = np.array(data)
    return data, labels




def load_images_from_csv(csv_file, start, end, label, desired_height, desired_width, image_folder):
    data = []
    labels = []

    df = pd.read_csv(csv_file)
    image_paths = df['filename'].tolist()[start:end]

    for filename in image_paths:
        filepath_full = os.path.join(image_folder, filename)
        if filepath_full.endswith(".png") and os.path.exists(filepath_full):
            image = skimage.io.imread(filepath_full)
            image = skimage.img_as_float32(image)
            #image = skimage.transform.resize(image, (desired_height, desired_width))
            data.append(image)
            labels.append(label)
    
    labels = np.array(labels)
    data = np.array(data)
    return data, labels

if __name__ == "__main__":
    pass