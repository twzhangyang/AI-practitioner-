import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

def detect_faces(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MTCNN detector
    detector = MTCNN()
    
    # Detect faces
    faces = detector.detect_faces(image_rgb)
    
    # Draw rectangles around the faces
    for face in faces:
        x, y, width, height = face['box']
        cv2.rectangle(image_rgb, (x, y), (x+width, y+height), (0, 255, 0), 2)
        
        # Optionally, draw key facial landmarks
        keypoints = face['keypoints']
        for point in keypoints.values():
            cv2.circle(image_rgb, point, 2, (0, 255, 0), 2)
    
    # Display the result using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    
    return faces

# Usage
image_path = 'image1.jpg'
detected_faces = detect_faces(image_path)
print(f"Number of faces detected: {len(detected_faces)}")

# Print additional information about each face
for i, face in enumerate(detected_faces):
    print(f"Face {i+1}:")
    print(f"  Box: {face['box']}")
    print(f"  Confidence: {face['confidence']:.2f}")
    print(f"  Keypoints: {face['keypoints']}")
    print()
