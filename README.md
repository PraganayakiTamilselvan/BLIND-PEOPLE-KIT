Project Overview
"Visual Assistance for Blind" is an innovative solution designed to assist visually impaired individuals.
This project combines object detection and handwritten note recognition to provide audio alerts, allowing users to understand their environment better and access handwritten information through sound.

Features
🔍 Object Detection: Detects various objects in the surroundings and provides real-time audio descriptions.

📝 Handwritten Notes to Audio: Captures handwritten notes and converts them into speech, enabling users to listen to the written content.

Technologies Used
Python

TensorFlow / OpenCV (for object detection and image processing)

Text-to-Speech (TTS) Engine (such as pyttsx3 or gTTS for audio alerts)

OCR (Optical Character Recognition) (such as Tesseract OCR for handwritten note recognition)

How it Works
Object Detection Module:

The system uses a camera to capture the surroundings.

Detected objects are identified and announced to the user via audio output.

Handwritten Note Recognition Module:

The user can point the camera at a handwritten note.

The note is scanned, recognized, and converted into an audible format for the user to hear.
