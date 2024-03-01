import cv2
import mediapipe as mp
import tkinter as tk
from threading import Thread
from PIL import Image, ImageTk

# Create a window
window = tk.Tk()

# Create a label for the camera frame
label = tk.Label(window)
label.pack()

# Create a button that will start the camera when clicked
button = tk.Button(window, text="Start Camera", command=lambda: Thread(target=start_camera).start())
button.pack()

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to start the camera
def start_camera():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        mp_face_detection = mp.solutions.face_detection
        face_regions = []  # Initialize face_regions

        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            blur_enabled = True
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip the image horizontally for a later selfie-view display
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to pass by reference.
                frame.flags.writeable = False
                results = hands.process(frame)

                # Draw the hand annotations on the image.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        hand_x, hand_y = hand_landmarks.landmark[0].x * frame.shape[1], hand_landmarks.landmark[0].y * frame.shape[0]
                        face_regions = detect_faces(frame, face_detection)
                        hand_in_face_region = False
                        for face_region in face_regions:
                            x1, y1, w, h = face_region
                            if x1 < hand_x < x1 + w and y1 < hand_y < y1 + h:
                                hand_in_face_region = True
                                break

                        if not hand_in_face_region:
                            finger_states, fingers_count = calculate_finger_states(hand_landmarks)

                            if fingers_count == 1:
                                blur_enabled = True
                            elif fingers_count == 5:
                                blur_enabled = False

                if blur_enabled:
                    frame = blur_faces(frame, face_regions)

                # Convert the image from OpenCV BGR format to Tkinter RGB format
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # Update the label with the new image
                label.config(image=image)
                label.image = image

    cap.release()

def detect_faces(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    face_regions = []
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = int(bbox.xmin * W), int(bbox.ymin * H), int(bbox.width * W), int(bbox.height * H)

            if w > 0 and h > 0:
                face_regions.append((x1, y1, w, h))

    return face_regions

def blur_faces(img, face_regions):
    for face_region in face_regions:
        x1, y1, w, h = face_region
        face_region = img[y1:y1 + h, x1:x1 + w, :]
        if not face_region.size == 0:
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(face_region, (30, 30))

    return img

def calculate_finger_states(hand_landmarks):
    finger_states = [0, 0, 0, 0, 0]

    if hand_landmarks.landmark[3].y > hand_landmarks.landmark[4].y:
        finger_states[0] = 1

    if hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y:
        finger_states[1] = 1

    if hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y:
        finger_states[2] = 1

    if hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y:
        finger_states[3] = 1

    if hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y:
        finger_states[4] = 1

    fingers_count = sum(finger_states)

    return finger_states, fingers_count

# Start the Tkinter event loop
window.mainloop()
