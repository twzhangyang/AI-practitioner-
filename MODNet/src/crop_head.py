import cv2
import numpy as np
import face_recognition

def crop_head_and_create_transparent(image_path, matte_path, face_locations, face_landmarks):
    # Load the original image and the matte
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    matte = cv2.imread(matte_path, cv2.IMREAD_GRAYSCALE)

    # Ensure matte is the same size as the image
    matte = cv2.resize(matte, (image.shape[1], image.shape[0]))

    for face_landmark in face_landmarks:
        # Get the bounding box of the face
        top = face_landmark['chin'][0][1]
        bottom = face_landmark['chin'][16][1]
        left = min(point[0] for point in face_landmark['chin'])
        right = max(point[0] for point in face_landmark['chin'])

        # Extend the bounding box to include the whole head
        top = max(0, top - int((bottom - top) * 0.5))  # Add 50% above the face
        bottom = min(image.shape[0], bottom + int((bottom - top) * 0.2))  # Add 20% below the chin
        left = max(0, left - int((right - left) * 0.2))  # Add 20% to the left
        right = min(image.shape[1], right + int((right - left) * 0.2))  # Add 20% to the right

        # Crop the head from the original image and the matte
        head_rgb = image_rgb[top:bottom, left:right]
        head_matte = matte[top:bottom, left:right]

        # Create a new image with transparent background
        head_rgba = np.zeros((bottom-top, right-left, 4), dtype=np.uint8)
        head_rgba[:,:,:3] = head_rgb
        head_rgba[:,:,3] = head_matte

        # Save the result
        cv2.imwrite(f"head_transparent_{left}_{top}.png", cv2.cvtColor(head_rgba, cv2.COLOR_RGBA2BGRA))

# Usage
image_path = "image1.jpg"
matte_path = "output_matte.png"
face_locations = face_recognition.face_locations(face_recognition.load_image_file(image_path))
face_landmarks = face_recognition.face_landmarks(face_recognition.load_image_file(image_path))

crop_head_and_create_transparent(image_path, matte_path, face_locations, face_landmarks)
