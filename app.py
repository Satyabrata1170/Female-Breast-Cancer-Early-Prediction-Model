from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load model
model = load_model('model.weights.best.keras')

# Create upload folder if not exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Adjust as per model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def preprocess_base64_image(data_url):
    header, encoded = data_url.split(",", 1)
    binary_data = base64.b64decode(encoded)
    img = Image.open(BytesIO(binary_data)).resize((128, 128))  # Adjust as needed
    img = img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)
        result = np.argmax(prediction)
        classes = ["Cancer", "Non-Cancer"]  # Adjust as per your training
        predicted_label = classes[result]

        return jsonify({'prediction': predicted_label, 'image_url': filepath})

    return jsonify({'error': 'Invalid file format'})

@app.route('/predict-camera', methods=['POST'])
def predict_camera():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'})

    try:
        img_array = preprocess_base64_image(data['image'])
        prediction = model.predict(img_array)
        result = np.argmax(prediction)
        classes = ["Cancer", "Non-Cancer"]
        predicted_label = classes[result]
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
