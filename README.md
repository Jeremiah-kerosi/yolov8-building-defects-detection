# Building Defects Detection with YOLOv8

This project is a web application that uses the YOLOv8 model to detect defects in uploaded images. The application is built using Flask and can handle image uploads, run object detection, and return results with bounding boxes drawn around detected defects.

## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Endpoints](#endpoints)
* [Model Details](#model-details)

## Installation

To run this project, you need to have Python installed. Follow the steps below to set up the environment and run the application.

1. Clone the repository:
    ```bash
    git clone https://github.com/Jeremiah-kerosi/yolov8-building-defects-detection.git
    cd yolov8-image-detection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the YOLOv8 model weights (`best.pt`) in the `yolov8-model` directory. If not, download or train the model and place the weights file in the directory.

4. Run the Flask application:
    ```bash
    python app.py
    ```

## Usage

1. Open your web browser and go to `http://127.0.0.1:5000/`.
2. Upload an image file in one of the allowed formats (`png`, `jpg`, `jpeg`, `gif`).
3. The application will process the image and display the results with detected objects highlighted.

## Project Structure

```markdown
.
├── app.py
├── requirements.txt
├── templates
│   └── index.html
├── uploads
├── yolov8-model
│   └── best.pt
└── utils
    └── model_utils.py
```
* `app.py`: The main Flask application.
* `requirements.txt`: List of Python packages required to run the application.
* `templates/index.html`: HTML template for the web interface.
* `uploads`: Directory for storing uploaded images.
* `yolov8-model/best.pt`: YOLOv8 model weights.
* `utils/model_utils.py`: Utility functions for model preprocessing.

## Endpoints

### `GET /`
Renders the main page for image upload.

### `POST /predict`
Handles image upload and object detection.

* **Request**: Expects a form-data with an image file.
* **Response**: JSON response containing:
  * `image_data`: Base64 encoded string of the image with bounding boxes.
  * `final_class`: The predicted class with the highest confidence score.
  * `width`: Original width of the uploaded image.
  * `height`: Original height of the uploaded image.

## Model Details

This application uses the YOLOv8 model for object detection. The model is loaded from the `yolov8-model/best.pt` file. Detection results include bounding box coordinates, class indices, and confidence scores.
