import pandas as pd
import numpy as np
import os
import pathlib
import librosa
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

labels, audio, files = [], [], []
DATA_PATH = "Calls" # lmk if this path doesn't work for you
# DATA_PATH = "/Users/bastien/Desktop/Spring 2023/CSS383/Calls"
# # need to change this later, just have it like this for dev purposes

def process_files() -> None:
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATA_PATH)):
        if dirpath is not DATA_PATH:
            # save label
            # dirpath had a backward slash so the labels weren't accurate
            dirpath = dirpath.split('\\') 
            labels.append(dirpath[-1])
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath[0], dirpath[1], f)
                files.append(file_path)
                # extract the signal audio (represents amplitude)
                # sample rate of the audio singal in Hz
                sig, sr = librosa.load(file_path, sr=22050)
                audio.append(sig)
            audio.append([-1])
    build_features()

def build_features() -> None:
    data_frame, idx = [], 0
    # audio features are too long/big to loop over
    for amp in audio:
        if (len(amp) == 1):
            idx += 1
        else:
            for signal in amp:
              data_frame.append(np.array([signal, labels[idx]]))

    # we can add more features here like frequency, MFCC etc...
    df = pd.DataFrame(data_frame, columns=["amplitude", "whale-names"])
    train_model(df)

def train_model(df: pd.DataFrame) -> float:
    features = df["amplitude"]
    label = df["whale-names"]

    # splitting the model
    features_train, features_test, label_train, label_test = \
       train_test_split(features, label, test_size = 0.2)

    model = DecisionTreeClassifier()
    model.fit(features_train, label_train)

    prediction_test = model.predict(features_test)
    return accuracy_score(label_test, prediction_test)

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

def main():
    process_files()

if (__name__ == "__main__"):
    main()
