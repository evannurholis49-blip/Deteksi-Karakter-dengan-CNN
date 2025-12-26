from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import io
import pickle

app = Flask(__name__)

# Load the trained model
with open('digit_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_digit(image):
    # Convert image to grayscale and resize
    image = image.convert('L').resize((28, 28))
    # Invert colors (MNIST digits are white on black)
    image = Image.eval(image, lambda x: 255 - x)
    # Convert to numpy array and flatten
    image_array = np.array(image).flatten() / 255.0
    # Predict
    prediction = model.predict([image_array])
    return int(prediction[0])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'})
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'error': 'No selected files'})
    predictions = []
    for file in files:
        if file.filename != '':
            try:
                # Read the image
                file_bytes = file.read()
                image = Image.open(io.BytesIO(file_bytes)).convert('L')  # Convert to grayscale
                image = image.resize((28, 28))  # Resize to 28x28

                # Make prediction
                predicted_digit = predict_digit(image)
                predictions.append(str(predicted_digit))
            except Exception as e:
                # If prediction fails, try to return a default or error message
                predictions.append('Tidak dapat diprediksi')
        else:
            predictions.append('File tidak valid')
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
