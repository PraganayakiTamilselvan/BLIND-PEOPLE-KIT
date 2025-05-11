import cv2
import time
import numpy as np
import serial
import requests
import os
from PIL import Image
import pytesseract
import pyttsx3
from darkflow.net.build import TFNet

# === Tesseract OCR config ===
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# === Text-to-speech setup ===
en = pyttsx3.init()
en.setProperty('rate', 80)

# === YOLO Setup ===
options = {
    'model': 'cfg/yolo.cfg',
    'load': 'cfg/yolov2.weights',
    'threshold': 0.3,
    'gpu': 0.7
}
tfnet = TFNet(options)

# === ESP32-CAM snapshot URL ===
ESP32_CAM_URL = "http://192.168.231.123/cam-hi.jpg"

# === Serial Setup ===
ser = serial.Serial('COM3', baudrate=9600, timeout=1)
ser.flushInput() 


def talk(text):
    en.say(text)
    en.runAndWait()


##def ocr_webcam():
##    cap = cv2.VideoCapture(0)
##    print("OCR mode started. Press 'q' to exit.")
##    while True:
##        ret, frame = cap.read()
##        if not ret:
##            print("Camera error.")
##            break
##        cv2.imwrite("temp.jpg", frame)
##        text = pytesseract.image_to_string(Image.open("temp.jpg"))
##        os.remove("temp.jpg")
##        if text.strip():
##            print("Text:", text)
##            talk(text)
##        else:
##            print("No text found.")
##        cv2.imshow("OCR Mode", frame)
##        if cv2.waitKey(1) & 0xFF == ord('q'):
##            break
##    cap.release()
##    cv2.destroyAllWindows()
def ocr_esp32():
    print("ESP32-CAM OCR mode. Press 'q' to exit.")
    while True:
        try:
            response = requests.get(ESP32_CAM_URL, timeout=2)
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)

            if frame is None:
                print("Failed to decode image from ESP32.")
                continue

            cv2.imshow("ESP32-CAM OCR", frame)

            # OCR
            cv2.imwrite("esp_temp.jpg", frame)
            text = pytesseract.image_to_string(Image.open("esp_temp.jpg"))
            os.remove("esp_temp.jpg")

            if text.strip():
                print("Text:", text)
                talk(text)
            else:
                print("No text found.")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"ESP32 OCR error: {e}")
            time.sleep(2)

    cv2.destroyAllWindows()




##def yolo_webcam():
##    cam = cv2.VideoCapture(0)
##    print("Webcam YOLO mode. Press 'q' to exit.")
##    while True:
##        ret, frame = cam.read()
##        if not ret:
##            print("Webcam error.")
##            break
##        output = tfnet.return_predict(frame)
##        for pred in output:
##            label = pred['label']
##            tl = (pred['topleft']['x'], pred['topleft']['y'])
##            br = (pred['bottomright']['x'], pred['bottomright']['y'])
##            color = (0, 255, 255)
##            cv2.rectangle(frame, tl, br, color, 2)
##            cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
##            print(f"Label: {label}")
##            talk(label)
##
##        cv2.imshow("Webcam YOLO", frame)
##        if cv2.waitKey(1) & 0xFF == ord('q'):
##            break
##    cam.release()
    cv2.destroyAllWindows()


def yolo_esp32():
    print("ESP32-CAM YOLO mode. Press 'q' to exit.")
    while True:
        try:
            response = requests.get(ESP32_CAM_URL, timeout=2)
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)

            if frame is None:
                print("Empty frame received.")
                continue

            output = tfnet.return_predict(frame)
            for pred in output:
                label = pred['label']
                tl = (pred['topleft']['x'], pred['topleft']['y'])
                br = (pred['bottomright']['x'], pred['bottomright']['y'])
                color = (0, 255, 0)
                cv2.rectangle(frame, tl, br, color, 2)
                cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                print(f"ESP32-CAM: {label}")
                talk(label)

            cv2.imshow("ESP32-CAM YOLO", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error fetching from ESP32-CAM: {e}")
            time.sleep(2)

    cv2.destroyAllWindows()


# === Main Serial Handler ===
print("Press button on hardware... (waiting for serial input)")
time.sleep(10)
if ser.in_waiting > 0:
    value = ser.readline().decode('ascii').strip()
    print(f"Received: {value}")

    if value == "1":
        ocr_esp32()
    elif value == "2":
        yolo_esp32()
    else:
        print("No valid command received.")
