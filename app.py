from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
import numpy as np
import base64

app = Flask(__name__)

# Load the pre-trained ML model
model_filename = 'freshness_detection_model.h5'
model = tf.keras.models.load_model(model_filename)

# Set the upload folder for images
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'})

    image_data = request.json['image']
    image_path = save_image(image_data)

    if isinstance(image_path, dict) and 'error' in image_path:
        # Log the error
        print(f"Error saving image: {image_path['error']}")
        return jsonify(image_path)

    try:
        prediction_score = predict_image(image_path)
        prediction_label = 'Spoiled' if prediction_score > 0.5 else 'Fresh'

        result = {
            'prediction_score': float(prediction_score),
            'prediction_label': prediction_label
        }

        return jsonify(result)
    except Exception as e:
        # Log the error
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f"Error during prediction: {str(e)}"})

def save_image(image_data):
    try:
        image_data = image_data.split(',')[1]  # Remove the 'data:image/jpeg;base64,' prefix
        image_data = bytes(image_data, encoding='utf-8')
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')

        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(image_data))

        return image_path
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return {'error': f"Failed to save image: {str(e)}"}

def predict_image(image_path):
    # Load and preprocess the input image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)

    return predictions[0][0]

if __name__ == '__main__':
    app.run(debug=True)
