import os
import base64
import traceback
import tempfile
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import glob

# ------------------------------
# CONFIG
# ------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # Reduced to 10MB for free tier

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Model configuration
MODELS_DIR = "models"
CLASS_NAMES = ["Bad Coffee Bean", "Good Coffee Bean"]

# Global variables for current model
current_model = None
current_model_name = None
current_model_type = None  # 'densenet' or 'custom'
current_img_size = 224

def get_available_models():
    """Get list of available model files in the models directory"""
    if not os.path.exists(MODELS_DIR):
        return []
    
    model_files = []
    # Look for .h5 and .keras files
    for ext in ['*.h5', '*.keras']:
        model_files.extend(glob.glob(os.path.join(MODELS_DIR, ext)))
    
    models = []
    for model_path in model_files:
        filename = os.path.basename(model_path)
        name = os.path.splitext(filename)[0]
        
        # Determine model type and image size
        if 'custom' in filename.lower():
            model_type = 'custom'
            img_size = 128
        else:
            model_type = 'densenet'
            img_size = 224
            
        models.append({
            'filename': filename,
            'name': name,
            'type': model_type,
            'img_size': img_size,
            'path': model_path
        })
    
    return models

def load_selected_model(model_path, model_type, img_size):
    """Load a specific model"""
    global current_model, current_model_name, current_model_type, current_img_size
    
    try:
        tf.keras.backend.clear_session()
        current_model = load_model(model_path, compile=False)
        current_model_name = os.path.basename(model_path)
        current_model_type = model_type
        current_img_size = img_size
        print(f"[INFO] Successfully loaded model: {model_path}")
        print(f"[INFO] Model type: {model_type}, Image size: {img_size}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        traceback.print_exc()
        return False

# Initialize with first available model
available_models = get_available_models()
print(f"[INFO] Found {len(available_models)} available models")

if available_models:
    first_model = available_models[0]
    print(f"[INFO] Loading first model: {first_model['name']}")
    load_selected_model(first_model['path'], first_model['type'], first_model['img_size'])
else:
    print("[WARNING] No models found. Please place model files (.h5 or .keras) in the 'models' directory")
    print("[INFO] Expected model directory structure:")
    print("  models/")
    print("    ├── coffee_densenet169.h5")
    print("    ├── coffee_custom.keras")
    print("    └── other_models.h5")

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image_densenet(img_path, target_size=(224, 224)):
    """Load an image file and preprocess it for DenseNet prediction"""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise Exception(f"Error processing image for DenseNet: {str(e)}")

def prepare_image_custom(img_path, target_size=(128, 128)):
    """Load an image file and preprocess it for Custom CNN prediction"""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # Custom CNN uses rescaling (normalize to [0,1])
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        raise Exception(f"Error processing image for Custom CNN: {str(e)}")

def prepare_image_pil_densenet(img, target_size=(224, 224)):
    """Prepare PIL image for DenseNet"""
    img = img.convert("RGB").resize(target_size)
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def prepare_image_pil_custom(img, target_size=(128, 128)):
    """Prepare PIL image for Custom CNN"""
    img = img.convert("RGB").resize(target_size)
    img_array = np.expand_dims(np.array(img), axis=0)
    # Custom CNN uses rescaling (normalize to [0,1])
    img_array = img_array / 255.0
    return img_array

def encode_image(img_path):
    """Convert image to base64 string for HTML display"""
    try:
        with open(img_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")

def interpret_prediction(pred, model_type):
    """Interpret model prediction based on model type"""
    if model_type == 'custom':
        # Custom CNN uses sigmoid activation (binary classification)
        # Output is a single value between 0 and 1
        prob = float(pred[0]) if hasattr(pred[0], '__iter__') else float(pred)
        if prob > 0.5:
            class_idx = 1  # Good Coffee Bean
            confidence = prob
        else:
            class_idx = 0  # Bad Coffee Bean
            confidence = 1.0 - prob
    else:
        # DenseNet uses softmax (categorical classification)
        class_idx = np.argmax(pred)
        confidence = float(np.max(pred))
    
    predicted_class = CLASS_NAMES[class_idx]
    return predicted_class, confidence * 100

# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def home():
    """Render the main page"""
    return render_template("index.html", 
                         available_models=available_models,
                         current_model_name=current_model_name,
                         current_img_size=current_img_size,
                         current_model_type=current_model_type)

@app.route("/api/models")
def get_models():
    """Get available models"""
    return jsonify({
        "models": available_models,
        "current_model": {
            "name": current_model_name,
            "type": current_model_type,
            "img_size": current_img_size
        }
    })

@app.route("/api/select-model", methods=["POST"])
def select_model():
    """Select and load a specific model"""
    data = request.get_json()
    model_filename = data.get('model_filename')
    
    if not model_filename:
        return jsonify({"error": "Model filename not provided"}), 400
    
    # Find the model in available models
    selected_model = None
    for model in available_models:
        if model['filename'] == model_filename:
            selected_model = model
            break
    
    if not selected_model:
        return jsonify({"error": "Model not found"}), 404
    
    # Load the selected model
    success = load_selected_model(
        selected_model['path'], 
        selected_model['type'], 
        selected_model['img_size']
    )
    
    if success:
        return jsonify({
            "success": True,
            "model": {
                "name": current_model_name,
                "type": current_model_type,
                "img_size": current_img_size
            }
        })
    else:
        return jsonify({"error": "Failed to load model"}), 500

from io import BytesIO
from PIL import Image

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image prediction requests (optimized for CPU and in-memory processing)"""
    if current_model is None:
        return jsonify({"error": "No model loaded. Please select a model first."}), 503

    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected"}), 400

    # Limit number of files for free tier
    if len(files) > 20:
        return jsonify({"error": "Maximum 20 files allowed per request"}), 400

    results = []
    batch_images = []
    valid_files = []

    # Determine target size based on current model
    target_size = (current_img_size, current_img_size)

    # Preprocess images in memory
    for f in files:
        if f.filename == "":
            continue

        filename = secure_filename(f.filename)
        if not filename or not allowed_file(filename):
            results.append({
                "filename": f.filename,
                "error": "Invalid file or unsupported type"
            })
            continue

        # Check file size
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0)
        if size > MAX_FILE_SIZE:
            results.append({
                "filename": filename,
                "error": "File too large (max 10MB)"
            })
            continue

        try:
            # Load image in memory
            img = Image.open(f).convert("RGB").resize(target_size)
            
            # Prepare image based on model type
            if current_model_type == 'custom':
                img_array = prepare_image_pil_custom(Image.open(f), target_size)
            else:
                img_array = prepare_image_pil_densenet(Image.open(f), target_size)
                
            batch_images.append(img_array)
            valid_files.append((filename, img))
        except Exception as e:
            results.append({
                "filename": filename,
                "error": f"Failed to process image: {str(e)}"
            })

    if not batch_images:
        return jsonify({
            "results": results,
            "total_files": len(files),
            "processed_files": 0,
            "success": False
        })

    # Stack images into one batch
    batch_array = np.vstack(batch_images)

    # Predict in one call (CPU-friendly)
    preds = current_model.predict(batch_array, verbose=0, batch_size=1)

    for i, (filename, img) in enumerate(valid_files):
        predicted_class, confidence = interpret_prediction(preds[i], current_model_type)

        # Encode image to base64 for frontend
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_encoded = base64.b64encode(buffered.getvalue()).decode()
        img_data = f"data:image/jpeg;base64,{img_encoded}"

        results.append({
            "filename": filename,
            "success": True,
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.1f}",
            "image": img_data
        })

    return jsonify({
        "results": results,
        "total_files": len(files),
        "processed_files": len(valid_files),
        "success": True,
        "model_info": {
            "name": current_model_name,
            "type": current_model_type,
            "img_size": current_img_size
        }
    })

@app.route("/api/model-info")
def model_info():
    """Return information about the loaded model"""
    return jsonify({
        "model_name": current_model_name,
        "model_loaded": current_model is not None,
        "model_type": current_model_type,
        "img_size": current_img_size,
        "class_names": CLASS_NAMES,
        "available_models": available_models
    })

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy" if current_model is not None else "degraded",
        "model_loaded": current_model is not None,
        "current_model": current_model_name,
        "available_models": len(available_models)
    })

# ------------------------------
# ERROR HANDLERS
# ------------------------------
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({"error": "File too large (max 10MB)"}), 413

@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    print(f"[ERROR] Server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(503)
def service_unavailable(e):
    """Handle service unavailable errors"""
    return jsonify({"error": "Service temporarily unavailable"}), 503

# ------------------------------
# PRODUCTION SETUP
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    
    print(f"[INFO] Starting Coffee Bean Quality Classifier")
    print(f"[INFO] Models directory: {MODELS_DIR}")
    print(f"[INFO] Available models: {len(available_models)}")
    for model in available_models:
        print(f"  - {model['name']} ({model['type']}, {model['img_size']}x{model['img_size']})")
    print(f"[INFO] Current model: {current_model_name} ({'loaded' if current_model else 'not loaded'})")
    print(f"[INFO] Current image size: {current_img_size}x{current_img_size}")
    print(f"[INFO] Running on port: {port}")
    print(f"[INFO] Debug mode: {debug_mode}")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)