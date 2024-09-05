import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_faces(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image file: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(image_rgb)
    face_landmarks = face_recognition.face_landmarks(image_rgb)
    
    # Draw rectangles around the faces
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)
    
    # Optionally, draw facial landmarks
    for landmarks in face_landmarks:
        for facial_feature in landmarks.values():
            for point in facial_feature:
                cv2.circle(image_rgb, point, 2, (0, 255, 0), 2)
    
    # Display the result using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    
    return face_locations, face_landmarks

# Usage
image_path = 'image1.jpg'
print("Detecting faces...")
face_locations, face_landmarks = detect_faces(image_path)
print(f"Number of faces detected: {len(face_locations)}")

# Print additional information about each face
for i, (top, right, bottom, left) in enumerate(face_locations):
    print(f"Face {i+1}:")
    print(f"  Location: Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left}")
    print(f"  Landmarks: {list(face_landmarks[i].keys())}")
    print()
