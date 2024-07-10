from audioprocessing import audio_to_spectogram_image

OUTPUT_PATH = ""
AUDIO_PATH=""
LABEL="male"

def save_spectogram_to_png(audio_path_files,file_output_path):
    audio_to_spectogram_image.save_segment_to_directory_gray_scale(audio_path_files,file_output_path)


save_spectogram_to_png(AUDIO_PATH,OUTPUT_PATH)