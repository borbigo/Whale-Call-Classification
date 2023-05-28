#!/bin/bash

# Create a virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

# to start flask
flask run