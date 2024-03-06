import cv2
import mediapipe as mp
import numpy as np

# Initialize objects to detect hands and eyes
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Loading models
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video stream from the camera
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand and face detection
    hands_results = hands.process(rgb_frame)
    face_mesh_results = face_mesh.process(rgb_frame)

    # Create a copy of the image to display control points
    points_frame = frame.copy()

    # Display hand control points
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(points_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    # Displaying facial control points
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(points_frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    # Display image with control points
    cv2.imshow('Points', points_frame)

    # Display video stream image
    cv2.imshow('Video', frame)

    # Interrupt by pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Resource release
cap.release()
cv2.destroyAllWindows()
