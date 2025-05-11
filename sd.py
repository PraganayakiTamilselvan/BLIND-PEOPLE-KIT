import cv2
import os
from PIL import Image
import pytesseract
import pyttsx3
import cv2
from time import sleep
import tensorflow as tf
import time
import numpy as np
from darkflow.net.build import TFNet


tf.disable_v2_behavior()

# Tesseract OCR setup
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/Tesseract.exe'

# Initialize text-to-speech engine
en = pyttsx3.init()
en.setProperty('rate', 80)

def talking_tom(text):
    en.say(text)
    en.runAndWait()



def headset():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not detected!")
        return

    print("OCR mode activated. Capturing text...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.imwrite("temp.jpg", frame)
        img = Image.open("temp.jpg")
        text = pytesseract.image_to_string(img)
        os.remove("temp.jpg")  # Cleanup temporary file

        if text.strip():
            print("Extracted Text:", text)
            en.say(text)
            en.runAndWait()
        else:
            print("No text detected.")

        cv2.imshow("Live OCR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting OCR mode...")
            break





def yolo():
    global ser
    import cv2
    import time
    import numpy as np
    from darkflow.net.build import TFNet
    ##from gtts import gTTS
    ##from playsound import playsound
    ##import os
    ##from time import sleep
    ##a=0


    options = {'model': 'cfg/yolo.cfg',
               'load': 'cfg/yolov2.weights',
               'threshold': 0.3,
               'gpu': 0.7}

    tfnet = TFNet(options)

    def yolo_solo(file_name):
        image_file = cv2.imread('images/' + file_name)
        output = tfnet.return_predict(image_file)

        for prediction in output:
            label = prediction['label']
            confidence = prediction['confidence']
            tl = (prediction['topleft']['x'], prediction['topleft']['y'])
            br = (prediction['bottomright']['x'], prediction['bottomright']['y'])
            color = tuple(255 * np.random.rand(3))
            image_file = cv2.rectangle(image_file, tl, br, color, 2)
            font = cv2.FONT_HERSHEY_COMPLEX
            image_file = cv2.putText(image_file, label, (tl[0], tl[1] - 5), font, .7, color, 2)

        output_file = 'output/' + file_name
        cv2.imwrite(output_file, image_file)
        cv2.imshow('YOLO: You Only Look Once', image_file)
        cv2.waitKey(0)
        return image_file


    def yolo_real_time():
        cam = cv2.VideoCapture(0)
        en = pyttsx3.init()
        en.setProperty('rate', 80)
        while True:
            tic = time.time()
            ret, frame = cam.read()
            output = tfnet.return_predict(frame)
            for prediction in output:
                label = prediction['label']
                confidence = prediction['confidence']
                tl = (prediction['topleft']['x'], prediction['topleft']['y'])
                br = (prediction['bottomright']['x'], prediction['bottomright']['y'])
                color = tuple(255 * np.random.rand(3))
                frame = cv2.rectangle(frame, tl, br, color, 2)
                font = cv2.FONT_HERSHEY_COMPLEX
                frame = cv2.putText(frame, label, tl, font, .7, color, 2)
                print(label)
                en.say(label)
                en.runAndWait()

##                if label == 'person':
##                    print('Person')
##                    ser.write('person'.encode())
                    
            cv2.imshow('YOLO in real time', frame)
            toc = time.time()
    ##        print('person'.format(1 / (toc - tic)))


            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()
        
            
            
    """ YOLO for single image """
    file_name = 'dog.jpg'
    
    """ YOLO for real time """
    yolo_real_time()

    print('All Done!')





def serial():
    global ser
    import serial
    ser = serial.Serial('COM14',baudrate='9600',timeout=1)
    ser.flushInput()
    print("press button with in 15 sec....")
    sleep(10)
    a1 = ser.readline().decode('ascii')
    print(a1)
    if(a1=="1"):
        headset()
    
    elif(a1=="2"):
        yolo()
    else:
       print("Not switch Click")
 
serial()        
