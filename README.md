# AI-Powered Fabric Defect Detection

A full-stack web application that uses Deep Learning (InceptionV3 & ResNet50) to detect and localize fabric defects. It features a secure user authentication system, an interactive dashboard, Grad-CAM visual explanations, and a Gemini-powered chatbot for quality control assistance.

## Features

- **Fabric Validation**: Automatically rejects non-fabric images using InceptionV3.
- **Defect Detection**: Classifies defects into 6 specific categories using ResNet50.
- **Visual Localization**: Uses Grad-CAM to assume exact defect locations and draws bounding boxes.
- **Explainable AI Chatbot**: Integrated Google Gemini API to explain defects and answer textile queries.
- **Secure Authentication**: User registration and login system.
- **History Tracking**: Keeps a record of all user predictions.

## Tech Stack

- **Backend**: Python, Flask, SQLite
- **AI/ML**: PyTorch, Torchvision, NumPy, OpenCV, Pillow
- **Frontend**: HTML5, CSS3 (Glassmorphism design), JavaScript
- **API**: Google Gemini (Generative AI)

## Installation & Setup

### 1. Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### 2. Clone the Repository
```bash
git clone <repository_url>
cd fabric-defect-detection
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
You need a Google Gemini API Key for the chatbot to function.
- Get a key from [Google AI Studio](https://aistudio.google.com/).
- Set it in your environment (PowerShell):
```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```
Or simply edit `app.py` line 44 (not recommended for production).

### 5. Run the Application
```bash
python app.py
```
The application will start at `http://127.0.0.1:5000`.

## Defect Classes
The system recognizes the following classes:
0. No defect
1. Thread defect 55
2. cut dataset
3. hole_dataset
4. stain_defect
5. thread_defects

## Project Structure
- `app.py`: Contains ALL backend logic (Routes, Models, DB, Chatbot).
- `models/`: Stores `.pth` model files.
- `static/`: Stores uploads, results, and CSS/JS assets.
- `templates/`: HTML frontend files.
- `requirements.txt`: Python dependencies.
