import librosa
import numpy as np

#dokończyć
def load_and_process_audio(start,end,folderpath,label,segment_length):

    data = []
    labels = []
    licznik=0

    for filename in os.listdir(folderpath)[start:end]:     
        filepath_full = os.path.join(folderpath, filename)
        y, sr = librosa.load(filepath_full)
        duration = segment_length
        segment_length = duration * sr
        segments = [y[i:i + segment_length] for i in range(0, len(y), segment_length)]
        for segment in segments:
            if len(segment) < segment_length:
                segment = np.tile(segment, int(np.ceil(segment_length / len(segment))))[:segment_length]             
            spectogram = create_spectogram(segment,sr)
        
            data.append(spectogram)
            labels.append(label)
    data = np.array(data)
    print("PL data: ",len(data))
    
    

def create_spectogram(y,sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.min)
    return S_DB