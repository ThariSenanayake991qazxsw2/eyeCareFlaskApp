from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('model/copy1ofm2_model.h5')

# Define the function to process the uploaded image
def process_image(image):
    img_size = 256
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(im_rgb, (img_size, img_size))
    data_new = resized / 255.0
    data_new = data_new.reshape(1, img_size, img_size, 3)
    return data_new

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            processed_image = process_image(image)
            prediction = model.predict(processed_image)
            result = np.argmax(prediction, axis=-1)
            if result == 0:
                prediction_text = 'Diabetes'
            elif result == 1:
                prediction_text = 'Normal'
            elif result == 2:
                prediction_text = 'Glaucoma'
            else:
                prediction_text = 'Cataract'
            return render_template('Prediction.html', prediction_text=prediction_text)
    return render_template('Prediction.html')

@app.route('/about')
def about():
    return render_template('About.html')

if __name__ == '__main__':
    app.run(debug=True)
