from flask import Flask, render_template, request
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import imutils
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import time
import os


app = Flask(__name__)

outputFrame = None
lock = threading.Lock()

my_dir = os.path.dirname(__file__)
mobile = os.path.join(my_dir, 'MobileNetV2.model')

maskNet = load_model(mobile)

prototxtPath = os.path.join(my_dir, "deploy.prototxt")
weightsPath = os.path.join(my_dir, "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)



vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detectFace(frame):
    # grab the dimensions of the frame and then construct a blob
	# from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    return (faceNet.forward(), w, h)

def getFaceLocs(detections, w, h, frame):
    faces = []
    locs = []
    preds = []

	# loop over the detections
    for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with
    # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > .75:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    return (faces, locs, preds)


def predictMask(faces, locs, preds, maskNet):
	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def detectMask():
    while True:
        global vs, outputFrame, lock
        # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=800)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
        (detections, w, h) = detectFace(frame)
        (faces, locs, preds) = getFaceLocs(detections, w, h, frame)
        (locs, preds) = predictMask(faces, locs, preds, maskNet)


	# loop over the detected face locations and their corresponding
	# locations
        for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            if label == "Mask":
                print("MASK", file=sys.stderr)
            else:
                print("NO MASK", file=sys.stderr)

		# include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            #timestamp = datetime.datetime.now()
		# display the label and bounding box rectangle on the output
		# frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        #with lock:
            #outputFrame = frame.copy()
			# show the output frame
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF
        with lock:
            outputFrame = frame.copy()



def generate():
    	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')



@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# loop over the frames from the video stream


if __name__ == "__main__":
    t = threading.Thread(target=detectMask)
    t.daemon = True
    t.start()
    app.run()