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

# ------------------------------
# CONFIG
# ------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # Reduced to 10MB for free tier

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Model configuration
MODEL_PATH = "models/coffee_densenet121.h5"
IMG_SIZE = 224
CLASS_NAMES = ["Bad Coffee Bean", "Good Coffee Bean"]

# Load model with error handling for production
model = None
try:
    if os.path.exists(MODEL_PATH):
        # Optimize for memory usage on free tier
        tf.keras.backend.clear_session()
        model = load_model(MODEL_PATH, compile=False)
        print(f"[INFO] Successfully loaded model: {MODEL_PATH}")
    else:
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    traceback.print_exc()

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path, target_size=(224, 224)):
    """Load an image file and preprocess it for DenseNet prediction"""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def encode_image(img_path):
    """Convert image to base64 string for HTML display"""
    try:
        with open(img_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")

def get_model_name():
    """Extract model name from path"""
    return os.path.basename(MODEL_PATH) if MODEL_PATH else "Unknown"

# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def home():
    """Render the main page"""
    return render_template("index.html", 
                         model_name=get_model_name(), 
                         img_size=IMG_SIZE)

from io import BytesIO
from PIL import Image

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image prediction requests (optimized for CPU and in-memory processing)"""
    if model is None:
        return jsonify({"error": "Model not available. Please try again later."}), 503

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
            img = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(np.array(img), axis=0)
            img_array = preprocess_input(img_array)
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
    preds = model.predict(batch_array, verbose=0, batch_size=1)

    for i, (filename, img) in enumerate(valid_files):
        class_idx = np.argmax(preds[i])
        predicted_class = CLASS_NAMES[class_idx]
        confidence = float(np.max(preds[i])) * 100

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
        "success": True
    })

@app.route("/api/model-info")
def model_info():
    """Return information about the loaded model"""
    return jsonify({
        "model_name": get_model_name(),
        "model_loaded": model is not None,
        "img_size": IMG_SIZE,
        "class_names": CLASS_NAMES
    })

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
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
    print(f"[INFO] Model path: {MODEL_PATH}")
    print(f"[INFO] Model loaded: {'Yes' if model is not None else 'No'}")
    print(f"[INFO] Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"[INFO] Running on port: {port}")
    print(f"[INFO] Debug mode: {debug_mode}")
    
    app.run(host="0.0.0.0", port=port, debug=debug_mode)