import os
import time
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras import backend as K
from PIL import Image
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Config for Production ---
# Use environment variables with fallbacks
PRODUCTION = os.environ.get('RENDER', False)

if PRODUCTION:
    app.config['UPLOAD_FOLDER'] = '/tmp'  # Render's temporary directory
    app.config['DEBUG'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
else:
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['DEBUG'] = True

app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['MODEL_PASSWORD'] = os.environ.get('MODEL_PASSWORD', 'secret123')

IMG_SIZE = 224

# Create directories only in development
if not PRODUCTION:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# --- Models ---
loaded_models = {}  # Cache for models if memory allows
DEFAULT_MODEL = 'coffee_densenet121.h5'
CLASS_LABELS = ['Bad Coffee Bean', 'Good Coffee Bean']

# Configure TensorFlow for production
if PRODUCTION:
    # Limit TensorFlow memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {e}")
    
    # Set thread limits for CPU
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def load_model(model_name):
    """Load model with error handling and caching logic"""
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # In production, don't cache models due to memory constraints
    if PRODUCTION:
        logger.info(f"Loading model {model_name} fresh for production...")
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    else:
        # In development, use caching
        if model_name not in loaded_models:
            logger.info(f"Loading and caching model {model_name}...")
            loaded_models[model_name] = tf.keras.models.load_model(model_path)
        return loaded_models[model_name]

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Preprocess image for DenseNet model"""
    try:
        img = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        arr = np.expand_dims(np.array(img), axis=0)
        arr = preprocess_input(arr.astype(np.float32))
        return arr
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        raise

def predict_image(image_path, model_name):
    """Make prediction on image"""
    model = None
    try:
        model = load_model(model_name)
        arr = preprocess_image(image_path)
        
        # Make prediction
        preds = model.predict(arr, verbose=0)  # Reduce logging output
        cls = int(np.argmax(preds[0]))
        conf = float(preds[0][cls])
        
        return cls, conf, None
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg
    finally:
        # Clean up memory after each prediction in production
        if PRODUCTION and model is not None:
            del model
            K.clear_session()
            tf.keras.backend.clear_session()

# --- Routes ---
@app.route('/')
def index():
    """Main page route"""
    try:
        model_files = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.h5')]
        return render_template('index.html', models=model_files, default_model=DEFAULT_MODEL)
    except Exception as e:
        logger.error(f"Error loading index page: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        model_name = request.form.get('model_name', DEFAULT_MODEL)
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files uploaded'}), 400

        results = []
        
        for i, file in enumerate(files):
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename, 
                    'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'
                })
                continue

            # Use temporary files for better memory management
            temp_file = None
            try:
                # Create secure temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=app.config['UPLOAD_FOLDER']) as temp_file:
                    file.save(temp_file.name)
                    temp_filepath = temp_file.name

                # Make prediction
                cls, conf, error = predict_image(temp_filepath, model_name)
                
                if error:
                    results.append({'filename': file.filename, 'error': error})
                else:
                    # Read image for base64 encoding
                    with open(temp_filepath, 'rb') as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    
                    results.append({
                        'filename': file.filename,
                        'success': True,
                        'predicted_class': CLASS_LABELS[cls] if cls < len(CLASS_LABELS) else f"Class {cls}",
                        'confidence': f"{conf * 100:.2f}%",
                        'confidence_raw': conf,
                        'image': f"data:image/jpeg;base64,{img_b64}",
                        'model_used': model_name
                    })
                    
            except Exception as e:
                error_msg = f"Error processing {file.filename}: {str(e)}"
                logger.error(error_msg)
                results.append({'filename': file.filename, 'error': error_msg})
            finally:
                # Clean up temporary file
                if temp_file and os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except Exception as e:
                        logger.warning(f"Could not remove temp file {temp_filepath}: {str(e)}")

        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        model_files = []
        if os.path.exists(app.config['MODEL_FOLDER']):
            model_files = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.h5')]
        
        return jsonify({
            'status': 'healthy',
            'environment': 'production' if PRODUCTION else 'development',
            'available_models': model_files,
            'cached_models': list(loaded_models.keys()) if not PRODUCTION else "No caching in production"
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug = not PRODUCTION
    
    logger.info(f"Starting app in {'production' if PRODUCTION else 'development'} mode on port {port}")
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug,
        threaded=True  # Enable threading for better performance
    )