import os
import cv2
import numpy as np

# Directory to save the dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Load the additional image
image_path = r'C:\Users\bhuvan\Desktop\as.png'  # Update the path to your image
additional_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if additional_image is None:
    print(f"Error: Could not load image from {image_path}.")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Resize the additional image to match the frame height
        H, W, _ = frame.shape
        additional_image_resized = cv2.resize(additional_image, (W, H))

        # Concatenate the video frame and the additional image side by side
        combined_frame = np.hstack((frame, additional_image_resized))

        cv2.putText(combined_frame, 'Ready? Press "Q" to start', (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', combined_frame)

        key = cv2.waitKey(25)
        if key == ord('q'):
            done = True
        elif key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            exit()

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Resize the additional image to match the frame height
        additional_image_resized = cv2.resize(additional_image, (W, H))

        # Concatenate the video frame and the additional image side by side
        combined_frame = np.hstack((frame, additional_image_resized))

        cv2.imshow('frame', combined_frame)
        key = cv2.waitKey(25)

        # Exit if the 'ESC' key is pressed
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()

