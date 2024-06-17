# ## Importing libraries

import os
import numpy as np
import io
from PIL import Image, ImageDraw
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import base64
from ultralytics import YOLO
from utils.model_utils import load_image_into_numpy_array

# ## Defining Constants

MODEL_PATH = "yolov8-model/best.pt"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ## Function to validate uploaded data

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ## Loading the trained model

model = YOLO(MODEL_PATH)

# ## Functions to handle routes

@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    detection_threshold = 0.5
    labels = ['crack', 'damaged roof', 'damaged paint']

    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            original_image = Image.open(file_path)
            original_width, original_height = original_image.size

            # Load and preprocess the image
            image_np = load_image_into_numpy_array(file_path)

            # Run the YOLOv8 model
            results = model.predict(source=image_np, conf=detection_threshold)

            # Extract detection data
            result = results[0]
            detection_boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates
            detection_classes = result.boxes.cls.numpy().astype(int)  # Class indices
            detection_scores = result.boxes.conf.numpy()  # Confidence scores

            # Draw bounding boxes and labels on the image
            image_with_boxes = Image.fromarray(image_np)
            draw = ImageDraw.Draw(image_with_boxes)
            for box, cls, score in zip(detection_boxes, detection_classes, detection_scores):
                if score > detection_threshold:
                    x_min, y_min, x_max, y_max = box
                    x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])
                    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
                    draw.text((x_min, y_min), f'{labels[cls]}: {score:.2f}', fill="red")

            # Convert PIL image to base64 encoded string
            buffered = io.BytesIO()
            image_with_boxes.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            img_str = img_str.decode('utf-8')

            # Clean up saved file
            os.remove(file_path)

            # Get the predicted class with the highest score
            highest_score_index = np.argmax(detection_scores)
            final_class = labels[detection_classes[highest_score_index]] if detection_scores[highest_score_index] > detection_threshold else "None"

            # Return the image data for AJAX request
            return jsonify({
                'image_data': img_str,
                'final_class': final_class,
                'width': original_width,
                'height': original_height
            })

    return 'No image uploaded or image type not allowed', 400

if __name__ == '__main__':
    app.run(debug=True)
