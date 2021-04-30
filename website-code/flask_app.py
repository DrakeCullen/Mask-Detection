import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys

app = Flask(__name__)

my_dir = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(my_dir, 'static/')

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

prototxtPath = os.path.join(my_dir, "deploy.prototxt")
weightsPath = os.path.join(my_dir, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNet(prototxtPath, weightsPath)


def maskDetection(imageURL):
    mobile = os.path.join(my_dir, 'MobileNetV2.model')
    model = load_model(mobile)
    image = cv2.imread(os.path.join(my_dir, 'static/' + imageURL))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > .6:
            print('Face!', file=sys.stderr)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]


            label = "Mask" if mask > withoutMask else "No Mask"
            if label == 'Mask':
                return 'Face Mask Detected!', round(mask * 100, 3)
            else:
                return 'No Mask Detected!', round(withoutMask * 100, 3)
        else:
            return "No Face Detected!", round((1 - confidence) * 100, 3)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		res, confidence = maskDetection(filename)
		flash(res)
		if res == 'Face Mask Detected!':
		    return render_template('mask.html', filename=filename, confidence=confidence)
		elif res == 'No Mask Detected!':
		    return render_template('nomask.html', filename=filename, confidence=confidence)
		else:
		    return render_template('noface.html', filename=filename, confidence=confidence)
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	res = maskDetection(filename)
	return redirect(url_for('static', filename=filename, res=res), code=301)

if __name__ == "__main__":
    app.run()