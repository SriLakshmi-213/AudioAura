import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygame
import librosa
import joblib
import logging
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize pygame and set up logging
pygame.init()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MusicNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MusicEmotionNet(MusicNet):
    def forward(self, x):
        x = super().forward(x)
        return torch.sigmoid(x)

def load_model(model_class, model_path, input_size, num_classes):
    logging.info(f"Loading model from {os.path.abspath(model_path)}")
    model = model_class(input_size, num_classes)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        logging.info(f"Model structure after loading:\n{model}")
    else:
        logging.warning(f"Model file not found: {model_path}. Initializing with random weights.")
    model.eval()
    return model

def extract_features(file_path, input_size):
    logging.info(f"Extracting features from {file_path}")
    audio, sr = librosa.load(file_path, duration=30)
    features = np.hstack((
        np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1),
        np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1),
        np.mean(librosa.feature.melspectrogram(y=audio, sr=sr), axis=1),
        np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1),
        np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr), axis=1)
    ))
    return features[:input_size]

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    
    print("\nAudio Controls:\nP - Play/Pause\nS - Stop\nQ - Quit playback and continue analysis")
    
    playing = True
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                pygame.mixer.music.stop()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    if playing:
                        pygame.mixer.music.pause()
                        playing = False
                        print("Paused")
                    else:
                        pygame.mixer.music.unpause()
                        playing = True
                        print("Resumed")
                elif event.key == pygame.K_s:
                    pygame.mixer.music.stop()
                    print("Stopped")
        
        if not pygame.mixer.music.get_busy() and playing:
            break
        
        clock.tick(10)

# Global variables
input_size = 166
num_genres = 39
num_emotions = 7
device = torch.device("cpu")

# Load models and encoders
genre_model = load_model(MusicNet, 'model_output/genre_model.pth', input_size, num_genres)
emotion_model = load_model(MusicEmotionNet, 'model_output/emotion_model.pth', input_size, num_emotions)

le_genre = joblib.load('model_output/genre_encoder.joblib') if os.path.exists('model_output/genre_encoder.joblib') else None
le_emotion = joblib.load('model_output/emotion_encoder.joblib') if os.path.exists('model_output/emotion_encoder.joblib') else None

genre_model.to(device)
emotion_model.to(device)

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_music():
    logging.info("Received file upload request")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = file.filename
        logging.info(f"Processing file: {filename}")
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        logging.info("Playing audio")
        play_audio(file_path)

        features = extract_features(file_path, input_size)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

        with torch.no_grad():
            genre_output = genre_model(features_tensor)
            emotion_output = emotion_model(features_tensor)

        _, predicted_genre = torch.max(genre_output.data, 1)
        predicted_genre = le_genre.inverse_transform(predicted_genre.cpu().numpy())[0] if le_genre else f"Genre_{predicted_genre.item()}"

        predicted_emotions = (emotion_output > 0.5).float().cpu().numpy()[0]
        predicted_emotions = le_emotion.inverse_transform(np.where(predicted_emotions)[0]) if le_emotion else [f"Emotion_{i}" for i in np.where(predicted_emotions)[0]]

        os.remove(file_path)  # Remove the uploaded file after analysis

        logging.info(f"Predicted genre: {predicted_genre}")
        logging.info(f"Predicted emotions: {predicted_emotions}")
        
        return jsonify({
            'genre': predicted_genre,
            'emotions': predicted_emotions,
            'message': f"Analysis complete! Genre: {predicted_genre}, Emotions: {', '.join(predicted_emotions)}"
        })

if __name__ == '__main__':
    app.run(debug=True)

pygame.quit()
