from keras.models import load_model
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_for_prediction = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_for_prediction = (image_for_prediction / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image_for_prediction)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove any leading/trailing whitespace
    confidence_score = prediction[0][index]

    # Draw the class name and confidence score on the image
    cv2.putText(image, f"Class: {class_name[2:]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image, f"{np.round(confidence_score * 100)}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
