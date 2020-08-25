import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
import argparse
import datetime
import time
import tensorflow as tf

cats_old = ["Standing","Fallen"]
cats = ["false","true"]

# prepare function for resizing the grayscale frame
def prepare(grayframe):
    imgsize = 125
    newarray = cv2.resize(grayframe,(imgsize*2,imgsize))
    return newarray.reshape(-1, imgsize*2, imgsize, 1)

# load trained model
faldet_model = tf.keras.models.load_model("FALL-64x3-CNN.model")

# getting the webcam setup
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

if args.get("video", None) is None:
    cap = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    cap = cv2.VideoCapture(args["video"])


firstFrame = None

falltext = "false"

while True:
    frame = cap.read()
    
    frame = frame if args.get("video", None) is None else frame[1]
    motiontext = "false"
    
    if frame is None:
        break
    
    # grayscale the frame and apply blur
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if firstFrame is None: 
        firstFrame = blurred_gray # create key frame, every other frame is compared to
        continue
    
    # 
    frameDelta = cv2.absdiff(firstFrame, blurred_gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # dilate the threshold image to fill in holes, then find contours on threshold image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = imutils.grab_contours(cnts)
    
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motiontext = "true"
        
        # model does predict here
        prediction = faldet_model.predict([prepare(gray)])
        falltext = cats[int(prediction[0][0])]
    
    
    # draw the text and timestamp on the frame
    cv2.putText(frame, f"Motion: {motiontext}", (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.putText(frame, f"Fallen: {falltext}", (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # show the frame and record if the user presses a key
    cv2.imshow("Fall Detection", frame)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
cap.stop() if args.get("video", None) is None else cap.release()
cv2.destroyAllWindows()

