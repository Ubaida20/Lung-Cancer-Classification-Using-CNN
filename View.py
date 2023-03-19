from flask import Blueprint, render_template, url_for, jsonify, request, send_from_directory
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

View = Blueprint( '_name_', "View")

names = [ "Malignant Case","Normal Case", "Bengin Case" ]

def predict_label(img_path):
    # Read image
    model = load_model("trained_model.h5")
    
    # Preprocess image
    image = cv2.imread(img_path)
    image = cv2.resize(image, (256, 256))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)
    output = names[np.argmax(pred)]
    print("Label", output)
    #labelName = names[label]
   # print("Label name:", labelName)
    return output

@View.route('/')
def Home():
    return render_template('home.html')

@View.route("/Prediction", methods=['GET', 'POST'])
def main():
	return render_template("Prediction.html")


@View.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("Prediction.html", prediction = p, img_path = img_path)