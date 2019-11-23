# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
from numpy import asarray, expand_dims
import imutils
import pickle
import time
import cv2
from PIL import Image
from keras.models import load_model
from keras import backend as K
# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = "face_detection/deploy.prototxt"
modelPath = "face_detection/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = load_model('facenet/facenet_keras.h5')
# load the actual face recognition model along with the label encoder
rec_path, le_path = "outputs/recognizer.pickle", "outputs/le.pickle"
recognizer = pickle.loads(open(rec_path, "rb").read())
le = pickle.loads(open(le_path, "rb").read())
conf_threshold = 0.5
required_size = (160, 160)
# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)		#0 for webcam or "video.mp4" for any video you guys have
time.sleep(2.0)
# start the FPS throughput estimator
fps = FPS().start()
# loop over frames from the video file stream

detected_poi= {}
while(cap.isOpened()):
    _, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with  the prediction
        confidence = detections[0, 0, i, 2]
		# filter out weak detections
        if confidence > conf_threshold:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
			# extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue
            image = Image.fromarray(face) 
            # resize pixels to the model size
            image = image.resize(required_size)
            face_array = asarray(image)
            face_pixels = face_array.astype('float32')
            # standardize pixel values across channels (global)
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean) / std
            # transform face into one sample
            samples = expand_dims(face_pixels, axis=0)
            # make prediction to get embedding
            K.clear_session()
            yhat = embedder.predict(samples)
            K.clear_session()
            vec = yhat[0]
            vec = np.reshape(vec, (1, 128))
            # perform classification to recognize the face
            K.clear_session()
            preds = recognizer.predict_proba(vec)[0]
            K.clear_session()
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            print(name)
                
            # draw the bounding box of the face along with the associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            if (name.lower() == "unknown"):
                #print("hello")
                continue
            else:  
                if name in detected_poi:
                    detected_poi[name] += 1
                else:
                    detected_poi.update({name:1})
                img=Image.fromarray(frame)
                img = img.save(name + str(detected_poi[name]))
            
            # update the FPS counter
            fps.update()
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):break
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
cv2.destroyAllWindows()                                                                                                          	       