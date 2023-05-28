import pandas as pd
import numpy as np
import os
import pathlib
import librosa as lib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, url_for, request, redirect

# Global access to the model.
model = None

def extract_features() -> None:
    """
    Loops through the whale directory and extracts the relevant features
    from the whale call, creating a dataframe of those features to train.
    """
    DATA_PATH = "Calls"
    features = []

    for whale in os.listdir(DATA_PATH):
        whale_path = os.path.join(DATA_PATH, whale)
        for whale_audio in os.listdir(whale_path):
            audio_path = os.path.join(whale_path, whale_audio)
            curr_features = build_features(audio_path)
            curr_mfcc = [mfcc for mfcc in curr_features[-1]]
            features.append([whale] + curr_features[:-1] + curr_mfcc)
    
    columns = ["Whale", "Centroid", "Bandwith", "Chroma", "RMS", "Flat", "Contrast", "ZCR"] + [f"MFCC-{i + 1}" for i in range(20)]
    df = pd.DataFrame(features, columns=columns)
    df.to_csv("audio_features.csv", index=False)

def build_features(file_path) -> list[float]:
    """
    All the relevant features extracted from the whale calls, including
    the spectral features, zero crossing rate, and MFCC.
    """
    signal, sample_rate = lib.load(file_path)
    centroid = lib.feature.spectral_centroid(y=signal, sr=sample_rate)
    bandwith = lib.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    chroma = lib.feature.chroma_cens(y=signal, sr=sample_rate)
    rms = lib.feature.rms(y=signal)
    flat = lib.feature.spectral_flatness(y=signal)
    contrast = lib.feature.spectral_contrast(S=np.abs(lib.stft(signal)), sr=sample_rate)
    zcr = lib.feature.zero_crossing_rate(y=signal)
    mfccs = lib.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=20)
    return [np.mean(centroid), np.mean(bandwith), np.mean(chroma), 
            np.mean(rms), np.mean(contrast), np.mean(flat), np.mean(zcr),
            np.mean(mfccs.T, axis=0)]

def train_model(df: pd.DataFrame) -> float:
    """
    Splitting the data into testing/training sets. Using a
    decision tree classifier to train the model and predict
    whale species.
    """
    global model
    features = df.loc[:, "Centroid":"MFCC-20"]
    label = df["Whale"]
    
    features_train, features_test, label_train, label_test = \
       train_test_split(features, label, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(features_train, label_train)

    prediction_test = model.predict(features_test)
    return accuracy_score(label_test, prediction_test)


def select_file(file_path: str) -> str:
    """
    Extracts the features from the audio file uploaded by the user
    and uses the trained model to make a predicted on the extracted
    features.
    """
    global model
    features = build_features(file_path)
    curr_mfcc = [mfcc for mfcc in features[-1]]
    os.remove(file_path)
    return model.predict([features[:-1] + curr_mfcc])[0]

def main():
    if (not os.path.exists("audio_features.csv")):
        extract_features()
    train_model(pd.read_csv("audio_features.csv"))

# Creates an instance of the flask web application
app = Flask(__name__)

@app.route("/")
def index():
    """
    When the website is loaded, it renders the HTML home page. It represents
    the main entry to our web application.
    """
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    A POST request for when a file is uploaded, and renders the results.html
    page to display the accuracy score and whale species.
    """
    f = request.files["file"]
    if (not f.filename.endswith(".wav")):
        return redirect(url_for("display_error", message="Please upload a .wav audio file!"))
    f.save(os.path.join("user_uploads", f.filename))
    return redirect(url_for("calculate", file_name=f.filename))


@app.route("/calculate/<file_name>")
def calculate(file_name: str):
    """
    If the user uploads a valid .wave file, then the features will be extracted
    from the uploaded audio and a prediction will be made. The accuracy score
    will be displayed as well as the name of species classified by the model.
    """
    species = select_file(os.path.join("user_uploads", file_name))
    return render_template("results.html", species=species)


@app.route("/display_error/<message>")
def display_error(message: str):
    """
    Rerenders the home page to display an error if the file upload
    was either not a .wave file or if no file was uploaded at all.
    """
    return render_template("index.html", message=message)


@app.route("/<path:path>")
def redirect_to_home(path: str):
    """
    If user tries to access the "results" page without uploading a 
    ile or if the user access non-existent page, it'll return the
    user back to the home page.
    """
    return redirect(url_for("index"))

if (__name__ == "__main__"):
    main()
    app.run(debug=True)