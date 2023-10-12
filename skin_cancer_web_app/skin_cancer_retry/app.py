# Flask
from flask import Flask, request, render_template,  jsonify
from tensorflow.keras.models import load_model
import cv2

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5002/')

# add your model_path
model_path = '/Users/Grampun/Desktop/skin_cancer_web_app/model/skinCancer-best-0.00006.hdf5'

# Load your own trained model
model = load_model(model_path)
model._make_predict_function()          
print('Model loaded. Start serving...')


def model_predict(img, model):

    collecter = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)

    resize_img = cv2.resize(collecter, (128, 128))
    final_img = np.array(resize_img)

    final_img = final_img.astype('float32')
    final_img /= 255

    extend_imgDims = np.expand_dims(final_img, axis=2)
    test_img = np.expand_dims(extend_imgDims, axis=0)

    index_id = model.predict(test_img)

    lable = np.argmax(index_id)

    return lable


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/articles')
def articles():
    return render_template('articles.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        preds = model_predict(img, model)

        if preds == 1:
            result = "Low Risk"
        else:
            result = "High Risk"

        return jsonify(result=result, probability=int(preds))

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)
