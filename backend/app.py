from flask import Flask, request, redirect, render_template, url_for
import os
import numpy as np
import pandas as pd
import librosa

app = Flask(__name__)
# Define the GUI function
@app.route('/Upload', methods=['POST'])
def uploader():
    """ Handles upload of audio file
    """
    f = request.files['file']
    print(f)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    return redirect(url_for('show_results', filename=f.filename))