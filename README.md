# 🎶 AudioAura

**AudioAura** is an AI-powered music analysis tool that detects genres and emotions in songs. Harnessing the power of machine learning, it scans audio files, classifies musical styles, and interprets emotional tones to provide insightful analyses. 

## 🌟 Features
- 🎧 Upload and analyze audio files for genre and emotion detection
- 💾 Save and load trained genre models for efficient reuse
- ⚡ Frontend built with React and Vite for fast development and deployment

## 🛠️ Tech Stack
- **Frontend**: React, Vite
- **Backend**: Python (Flask or FastAPI), ML libraries (Librosa, TensorFlow/PyTorch)
- **Deployment**: Vite for frontend build and hot module replacement

## 📥 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo-url
    cd AudioAura
    ```

2. Install frontend dependencies:
    ```bash
    npm install
    ```

## 🏗️ Training and Using Models

1. Navigate to the `my-react-app` directory and then to the `ml_code` folder:
    ```bash
    cd my-react-app/ml_code
    ```

2. Train your model using the provided Python script:
    ```bash
    python3 train_models.py
    ```
   This will save the trained models to the specified path.

3. Use the saved genre paths in your frontend for real-time music analysis.

## 🚀 Running the Backend

1. After training the models, run the real-time analysis script:
    ```bash
    python3 realtime_analysis.py
    ```

## 🌐 Running the Frontend

1. Open another terminal and navigate back to the `my-react-app` directory:
    ```bash
    cd my-react-app
    ```

2. Start the frontend:
    ```bash
    npm start
    ```

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the project! Your contributions are welcome!

## 📄 License

This project is licensed under the [MIT License](LICENSE).
