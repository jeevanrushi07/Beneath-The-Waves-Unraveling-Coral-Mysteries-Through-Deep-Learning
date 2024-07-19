from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

model_path = 'waves_img.h5'
model = load_model(model_path)

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(299, 299))  # Use the correct target size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image

    prediction = model.predict(img_array)
    return prediction

@app.route('/output', methods=['GET', 'POST'])
def output():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'static/uploads', f.filename)
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        f.save(filepath)

        # Make prediction
        prediction = model_predict(filepath, model)
        index = ['Bleached Coral', 'Healthy Coral']
        result = index[int(np.argmax(prediction, axis=1))]

        return render_template("output.html", predict=result, img_path=f'/static/uploads/{f.filename}')
    return render_template("output.html")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    return render_template('inner-page.html')

if __name__ == '__main__':
    app.run(debug=True, port=2222)
