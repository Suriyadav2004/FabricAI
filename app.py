import os
import sqlite3
import datetime
import time
import logging
from functools import wraps

import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, g
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import google.generativeai as genai

# ==============================================================================
# CONFIGURATION
# ==============================================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super_secret_fabric_key_12345')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit
app.config['DATABASE'] = 'fabric_app.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set - chatbot will be disabled")

# ==============================================================================
# CONSTANTS & MAPPINGS
# ==============================================================================
DEFECT_CLASSES = {
    0: "No defect",
    1: "Thread defect 55",
    2: "cut dataset",
    3: "hole_dataset",
    4: "stain_defect",
    5: "thread_defects"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ==============================================================================
# PREPROCESSING TRANSFORMS
# ==============================================================================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def transform_pil_image(pil_img):
    return preprocess(pil_img).unsqueeze(0).to(DEVICE)

# ==============================================================================
# DATABASE
# ==============================================================================
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                result_path TEXT NOT NULL,
                defect_class_id INTEGER NOT NULL,
                defect_class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        db.commit()

init_db()

# ==============================================================================
# IMPROVED GRAD-CAM
# ==============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()

        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam

def generate_defect_visualization(image_path, model, predicted_class):
    orig_img = Image.open(image_path).convert("RGB")
    orig = np.array(orig_img)
    h, w = orig.shape[:2]

    input_tensor = transform_pil_image(orig_img)

    grad_cam = GradCAM(model, model.layer4[-1])
    cam = grad_cam(input_tensor, predicted_class)

    # Resize CAM to original image size for better visualization
    cam = cv2.resize(cam, (w, h))
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.5 + orig * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    timestamp = str(int(time.time()))
    filename = secure_filename(os.path.basename(image_path))
    result_filename = f"result_{timestamp}_{filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    
    cv2.imwrite(result_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return result_path

# ==============================================================================
# MODEL LOADING (using your renamed files: model1.pth and model2.pth)
# ==============================================================================
MODELS = {"fabric_validator": None, "defect_classifier": None}

def load_models():
    logger.info("=== LOADING MODELS ===")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Fabric Validator (model1.pth) - Optional
    val_path = os.path.join(base_dir, "models", "model1.pth")
    if os.path.exists(val_path):
        try:
            # Load checkpoint first to determine the number of classes
            checkpoint = torch.load(val_path, map_location=DEVICE)
            state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('state_dict', checkpoint)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            # Determine number of classes from the saved weights
            num_classes = state_dict['fc.weight'].shape[0]  # This should be 1 for binary classification
            
            # Initialize model with the correct number of classes
            # Based on the training code, this is a binary classifier with 1 output neuron
            model_v3 = models.inception_v3(weights=None, num_classes=num_classes, aux_logits=True)
            
            # Load the state dict
            model_v3.load_state_dict(state_dict, strict=True)
            
            # If we need to change the number of classes, do it after loading
            # But for now, we'll keep it as is since it matches the training
            
            # Disable aux_logits for inference
            model_v3.aux_logits = False
            
            model_v3.to(DEVICE)
            model_v3.eval()
            MODELS["fabric_validator"] = model_v3
            logger.info("✅ Fabric Validator loaded (model1.pth)")
        except Exception as e:
            logger.error(f"❌ Fabric Validator failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("model1.pth not found - fabric validation skipped")

    # 2. Defect Classifier (model2.pth) - REQUIRED
    defect_path = os.path.join(base_dir, "models", "model2.pth")
    if not os.path.exists(defect_path):
        logger.error(f"❌ DEFECT CLASSIFIER NOT FOUND: {defect_path}")
        logger.error("   Please place 'model2.pth' in the 'models/' folder")
        return

    try:
        model_r50 = models.resnet50(weights=None)
        # Handle different model architectures
        # The model has sequential layers in the fc layer
        model_r50.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model_r50.fc.in_features, 6)
        )

        checkpoint = torch.load(defect_path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model_r50.load_state_dict(state_dict)
        model_r50.to(DEVICE)
        model_r50.eval()
        MODELS["defect_classifier"] = model_r50
        logger.info("✅ Defect Classifier loaded successfully (model2.pth)")
    except Exception as e:
        logger.error(f"❌ DEFECT CLASSIFIER LOAD FAILED: {e}")
        import traceback
        traceback.print_exc()

load_models()

# ==============================================================================
# AUTH & ROUTES (unchanged - already perfect)
# ==============================================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            db = get_db()
            try:
                db.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                           (username, generate_password_hash(password)))
                db.commit()
                flash('Registered successfully! Please login.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username already exists.', 'error')
        else:
            flash('Fill all fields.', 'error')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=session.get('username'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if ext not in {'png', 'jpg', 'jpeg', 'bmp', 'gif'}:
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    filename = f"{timestamp}_{filename}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    return jsonify({
        "status": "success",
        "filepath": path,
        "url": url_for('static', filename=f'uploads/{filename}')
    })

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.get_json()
    filepath = data.get('filepath') if data else None
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Image not found"}), 400

    try:
        validator = MODELS["fabric_validator"]
        if validator:
            input_tensor = transform_pil_image(Image.open(filepath).convert("RGB"))
            with torch.no_grad():
                output = validator(input_tensor)
                # For binary classifier with 1 output neuron, check if output > 0
                # (or apply sigmoid and check if > 0.5)
                if isinstance(output, tuple):
                    # If it's a tuple (main_output, aux_output), use the main output
                    pred_prob = torch.sigmoid(output[0]).item()
                else:
                    # Otherwise, use the output directly
                    pred_prob = torch.sigmoid(output).item()
                
                # If probability of being fabric is less than 0.5, reject
                if pred_prob < 0.5:
                    return jsonify({"status": "rejected", "message": "Not a fabric image"})

        classifier = MODELS["defect_classifier"]
        if not classifier:
            return jsonify({"error": "Defect classifier not loaded"}), 500

        input_tensor = transform_pil_image(Image.open(filepath).convert("RGB"))
        with torch.no_grad():
            output = classifier(input_tensor)
            probs = F.softmax(output, dim=1)[0]
            confidence, pred_id = torch.max(probs, 0)
            class_name = DEFECT_CLASSES.get(pred_id.item(), "Unknown")

        result_path = generate_defect_visualization(filepath, classifier, pred_id.item())

        db = get_db()
        db.execute('''
            INSERT INTO predictions 
            (user_id, image_path, result_path, defect_class_id, defect_class_name, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], filepath, result_path, pred_id.item(), class_name,
              confidence.item(), datetime.datetime.now().isoformat()))
        db.commit()

        return jsonify({
            "status": "success",
            "class_name": class_name,
            "confidence": f"{confidence.item():.2%}",
            "result_url": url_for('static', filename=f'results/{os.path.basename(result_path)}'),
            "original_url": url_for('static', filename=f'uploads/{os.path.basename(filepath)}')
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Processing failed"}), 500

@app.route('/history')
@login_required
def history():
    rows = get_db().execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 50",
        (session['user_id'],)
    ).fetchall()

    data = [{
        "image": url_for('static', filename=f'results/{os.path.basename(r["result_path"])}'),
        "defect": r["defect_class_name"],
        "confidence": f"{r['confidence']:.2%}",
        "date": r["timestamp"].split('T')[0]
    } for r in rows]

    return jsonify(data)

@app.route('/chatbot', methods=['POST'])
@login_required
def chatbot():
    data = request.get_json()
    message = data.get('message', '').strip() if data else ''
    context = data.get('currentDefect')

    if not message:
        return jsonify({"reply": "Please send a message"}), 400
    if not GEMINI_API_KEY:
        return jsonify({"reply": "Chatbot not configured"}), 500

    try:
        prompt = (
            "You are FabricAI, a friendly textile quality expert. "
            "Answer concisely and professionally."
        )
        if context:
            prompt += f" The current defect is '{context}'."
        prompt += f"\nUser: {message}"

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return jsonify({"reply": response.text})
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return jsonify({"reply": "AI assistant temporarily unavailable"})

@app.errorhandler(404)
def not_found(e):
    return render_template('landing.html'), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)