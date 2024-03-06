import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceLandmarks

#load
fl = FaceLandmarks()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
while ret:
    frame = cv2.resize(frame, (640, 480))
    frame_copy = frame.copy()
    height, width, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = face_mesh.process(img_rgb)
    #1 face landmark detection
    if out.multi_face_landmarks is not None:
        # Get the facial landmarks
        landmarks = fl.get_facial_landmarks(frame)
        if landmarks is not None and landmarks.ndim == 2 and landmarks.shape[1] == 2:
            convexhull = cv2.convexHull(landmarks)
            #2 face blurrying
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, convexhull, 255)

            # extract the face
            frame_copy = cv2.blur(frame_copy, (30, 30))
            face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

            # extract background
            background_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(frame, frame, mask=background_mask)

            # final result
            result = cv2.add(background, face_extracted)
        else:
            result = frame
    else:
        result = frame

    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()