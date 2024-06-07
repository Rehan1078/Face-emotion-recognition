import cv2
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array


# Load the model from the JSON file
json_file = open("my_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("my_model.h5")

# Load the Haar cascade file for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define labels
labels = {0: 'angry',
          1: 'disgust',
          2: 'fear',
          3: 'happy',
          4: 'neutral',
          5: 'sad',
          6: 'surprise'}

# Function to preprocess the image for prediction
def preprocess_image(image):
    image = cv2.resize(image, (48, 48))  # Resize the image to 48x48
    image = img_to_array(image)  # Convert the image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

# Initialize webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # Region of interest (face)
        preprocessed_image = preprocess_image(roi_gray)  # Preprocess the face region
        pred = model.predict(preprocessed_image)  # Predict the emotion
        prediction_label = labels[pred.argmax()]  # Get the label for the highest confidence score
        
        # Draw rectangle around the face and add the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, prediction_label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the output frame
    cv2.imshow("Face Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
