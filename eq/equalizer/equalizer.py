import os
import librosa
import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def process_audio(audio_path, bass_gain, midrange_gain, treble_gain):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Apply equalizer settings
    y_bass = librosa.effects.preemphasis(y, coef=bass_gain)
    y_midrange = librosa.effects.preemphasis(y, coef=midrange_gain)
    y_treble = librosa.effects.preemphasis(y, coef=treble_gain)

    # Combine channels
    y_combined = (y_bass + y_midrange + y_treble) / 3.0

    return y_combined, sr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/equalizer', methods=['POST'])
def equalizer():
    # Get uploaded audio file and equalizer settings
    audio_file = request.files['audio']
    bass_gain = float(request.form['bass'])
    midrange_gain = float(request.form['midrange'])
    treble_gain = float(request.form['treble'])

    # Save uploaded audio file
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(audio_path)

    # Process audio
    processed_audio, sr = process_audio(audio_path, bass_gain, midrange_gain, treble_gain)

    # Save processed audio
    processed_audio_path = os.path.join(PROCESSED_FOLDER, audio_file.filename.split('.')[0] + '.wav')
    sf.write(processed_audio_path, processed_audio, sr)

    # Serve processed audio for download
    return send_file(processed_audio_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
