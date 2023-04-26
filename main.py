import pandas as pd
import numpy as np
import os
import pathlib
import librosa

labels = []
audio = []
files = []
DATA_PATH = "/Users/bastien/Desktop/Spring 2023/CSS383/Calls" # need to change this later, just have it like this for dev purposes

for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATA_PATH)):
    if dirpath is not DATA_PATH:
        # save label
        dirpath = dirpath.split("/") 
        label = dirpath[-1]
        
        for f in filenames:
            files.append(f)
            
            # load audio file
            file_path = os.path.join(dirpath, f)
            sig, sr = librosa.load(file_path, sr = 22050)
            audio.append(sig)
            
print(labels)

''' from 484 project
features = []
for i, file_dir in enumerate(file_dirs):
    label = labels[i]
    files = os.listdir(file_dir)
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(file_dir, file)
            features.append([feature, label, file])
'''
