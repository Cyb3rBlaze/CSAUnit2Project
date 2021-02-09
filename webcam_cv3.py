import cv2
import sys
import datetime as dt
from time import sleep

#Link to the model file with pretrained weights and biases
cascPath = "model.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#Creating video object to interact with webcam and dispaly
video_capture = cv2.VideoCapture(0)
anterior = 0

#Overlay image which replaces detected face
image = cv2.imread("arin.png")

#Saved values to prevent rigid transitions and flow smoothly
x_real_saved = 0
y_real_saved = 0
width_real_saved = 0
height_real_saved = 0

#While loop that iterates as fast as possible until program is killed
while True:
    #Condition if the camera has not been opened by OpenCV
    if not video_capture.isOpened():
        print('Cannot load camera.')
        sleep(5)
        pass

    #Detecting the actual image data returned by the webcam
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detecting the bounding boxes returned by the object detection model
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(40, 40))

    #The actual values of the bounding box for pasting the overlay
    x_real = 0
    y_real = 0
    width_real = 0
    height_real = 0

    #Extracting data from returned object of the OD model
    for (x, y, w, h) in faces:
        x_real = x
        y_real = y
        width_real = w
        height_real = h

    if anterior != len(faces):
        anterior = len(faces)

    #Expanding the overlay slightly to make it fit on the face better
    x_real = x_real-50
    y_real = y_real-50

    width_real += 100
    height_real += 100

    #Constraints to prevent bad dimensions and coordinate interaction
    if width_real < 10:
        width_real = 10
    if height_real < 10:
        height_real = 10
    if x_real < 0:
        x_real = 0
    if y_real < 0:
        y_real = 0
    
    #If they are bad use the saved values which are global relative to the local values in the while loop
    if x_real == 0:
        x_real = x_real_saved
        y_real = y_real_saved
        width_real = width_real_saved
        height_real = height_real_saved
    elif x_real != 0 and y_real != 0 and width_real != 10 and height_real != 10:
        x_real_saved = x_real
        y_real_saved = y_real
        width_real_saved = width_real
        height_real_saved = height_real

    #Resizes the image based on the bounding box dimensions
    resized = cv2.resize(image, (width_real, height_real), interpolation = cv2.INTER_AREA)

    #Replaces the ares of the image with the overlay
    frame[y_real:y_real+height_real, x_real:x_real+width_real] = resized

    #Displaying the final image to the user every iteration
    cv2.imshow('Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Prevents usage of resources after the program is killed
video_capture.release()
cv2.destroyAllWindows()
