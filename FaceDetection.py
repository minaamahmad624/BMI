import cv2
import dlib
import matplotlib.pyplot as plt
import imutils
from imutils import face_utils

# Initialize Dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the input image and convert it to grayscale
image = cv2.imread("./images/image38.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray, 1)

# Loop over the face detections
for (i, face) in enumerate(faces):
    # Determine the facial landmarks for the face region
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)  # Convert to NumPy Array

    # Draw the face bounding box
    (x, y, w, h) = face_utils.rect_to_bb(face)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Loop over the (x, y)-coordinates for the facial landmarks and draw them
    for (x_point, y_point) in shape:
        cv2.circle(image, (x_point, y_point), 1, (0, 0, 255), -1)

# Convert BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the annotated image
plt.figure(figsize=(8, 8))
plt.imshow(image_rgb)
plt.axis('off')
plt.title('Annotated Facial Landmarks')
plt.show()
