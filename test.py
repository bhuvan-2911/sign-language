import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Load the additional image
image_path = r'C:\Users\bhuvan\Desktop\as.png'
additional_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if additional_image is None:
    print(f"Error: Could not load image from {image_path}.")
    exit()

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    if frame is None:
        print("Error: Frame is None.")
        break

    H, W, _ = frame.shape

    # Resize the additional image to match the frame height (for side-by-side display)
    additional_image_resized = cv2.resize(additional_image, (W, H))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw on
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)

            min_x = min(x_)
            min_y = min(y_)
            max_x = max(x_)
            max_y = max(y_)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            # Ensure the data_aux has the correct number of features (42)
            if len(data_aux) == 42:
                x1 = int(min_x * W) - 10
                y1 = int(min_y * H) - 10
                x2 = int(max_x * W) - 10
                y2 = int(max_y * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            else:
                print(f"Unexpected number of features: {len(data_aux)} (expected 42)")

    # Concatenate the original frame and the additional image side by side
    combined_frame = np.hstack((frame, additional_image_resized))

    # Display the exit message on the combined frame
    cv2.putText(combined_frame, "Press 'ESC' to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', combined_frame)

    # Check if the 'ESC' key was pressed
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ESC key code
        break

cap.release()
cv2.destroyAllWindows()

