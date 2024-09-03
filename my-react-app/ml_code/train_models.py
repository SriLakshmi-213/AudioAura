import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Feature extraction for audio
def extract_audio_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, duration=30)
        
        # Extract various audio features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        
        # Combine features
        features = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz, axis=1)
        ))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Neural network for classification
class MusicNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Function to prepare data
def prepare_data(data_dir):
    X = []
    y_genre = []
    y_emotion = []

    print(f"Looking for audio files in: {data_dir}")

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.mp3', '.wav')):
                file_path = os.path.join(root, file)
                features = extract_audio_features(file_path)
                if features is not None:
                    X.append(features)
                    # Extract genre and emotion from directory structure
                    genre = os.path.basename(os.path.dirname(file_path))
                    emotion = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                    y_genre.append(genre)
                    y_emotion.append(emotion)
                    print(f"Processed: {file_path}")

    return np.array(X), np.array(y_genre), np.array(y_emotion)

# Function to train model
def train_model(model, train_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Main execution
if __name__ == "__main__":
    # Set up parameters
    data_dir = 'add the folder containing music with specific genres'
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    X, y_genre, y_emotion = prepare_data(data_dir)

    if len(X) == 0:
        raise ValueError("No valid audio files were processed. Check the data directory and file formats.")

    # Encode labels
    le_genre = LabelEncoder()
    y_genre_encoded = le_genre.fit_transform(y_genre)

    le_emotion = LabelEncoder()
    y_emotion_encoded = le_emotion.fit_transform(y_emotion)

    # Split the data
    X_train, X_test, y_genre_train, y_genre_test, y_emotion_train, y_emotion_test = train_test_split(
        X, y_genre_encoded, y_emotion_encoded, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_genre_train_tensor = torch.LongTensor(y_genre_train)
    y_genre_test_tensor = torch.LongTensor(y_genre_test)
    y_emotion_train_tensor = torch.LongTensor(y_emotion_train)
    y_emotion_test_tensor = torch.LongTensor(y_emotion_test)

    # Create datasets and dataloaders
    train_dataset_genre = TensorDataset(X_train_tensor, y_genre_train_tensor)
    test_dataset_genre = TensorDataset(X_test_tensor, y_genre_test_tensor)
    train_loader_genre = DataLoader(train_dataset_genre, batch_size=32, shuffle=True)
    test_loader_genre = DataLoader(test_dataset_genre, batch_size=32, shuffle=False)

    train_dataset_emotion = TensorDataset(X_train_tensor, y_emotion_train_tensor)
    test_dataset_emotion = TensorDataset(X_test_tensor, y_emotion_test_tensor)
    train_loader_emotion = DataLoader(train_dataset_emotion, batch_size=32, shuffle=True)
    test_loader_emotion = DataLoader(test_dataset_emotion, batch_size=32, shuffle=False)

    # Initialize models
    input_size = X_train.shape[1]
    num_genres = len(np.unique(y_genre))
    num_emotions = len(np.unique(y_emotion))

    genre_model = MusicNet(input_size, num_genres).to(device)
    emotion_model = MusicNet(input_size, num_emotions).to(device)

    # Define loss functions and optimizers
    criterion = nn.CrossEntropyLoss()
    genre_optimizer = optim.Adam(genre_model.parameters(), lr=0.001)
    emotion_optimizer = optim.Adam(emotion_model.parameters(), lr=0.001)

    # Train models
    print("Training Genre Model...")
    train_model(genre_model, train_loader_genre, genre_optimizer, criterion, num_epochs, device)

    print("Training Emotion Model...")
    train_model(emotion_model, train_loader_emotion, emotion_optimizer, criterion, num_epochs, device)

    print("Training completed.")

    # Save the models and encoders
    output_dir = './model_output'
    os.makedirs(output_dir, exist_ok=True)

    genre_model_path = os.path.join(output_dir, 'genre_model.pth')
    emotion_model_path = os.path.join(output_dir, 'emotion_model.pth')
    genre_encoder_path = os.path.join(output_dir, 'genre_encoder.joblib')
    emotion_encoder_path = os.path.join(output_dir, 'emotion_encoder.joblib')

    torch.save(genre_model.state_dict(), genre_model_path)
    torch.save(emotion_model.state_dict(), emotion_model_path)
    joblib.dump(le_genre, genre_encoder_path)
    joblib.dump(le_emotion, emotion_encoder_path)

    print(f"Genre model saved to: {genre_model_path}")
    print(f"Emotion model saved to: {emotion_model_path}")
    print(f"Genre encoder saved to: {genre_encoder_path}")
    print(f"Emotion encoder saved to: {emotion_encoder_path}")
