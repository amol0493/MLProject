from flask import Flask, request, jsonify, render_template
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)

# Get the path to the haarcascade_frontalface_default.xml file
cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

def contains_person(image):
    img = np.array(Image.open(BytesIO(image)).convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        return True, 100.0  # Confidence score not available with Haar Cascade
    else:
        return False, 0.0

@app.route('/')
def index():
    return render_template('shashi_sir.html')

@app.route('/contains_person', methods=['POST'])
def contains_person_endpoint():
    try:
        image = request.files['image'].read()
        result, confidence_score = contains_person(image)

        return jsonify({'result': result, 'confidence_score': confidence_score, 'message': 'Person detected' if result else 'No person detected'})

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
