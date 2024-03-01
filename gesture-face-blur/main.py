import os
import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceLandmarks

# Initialize objects to detect hands and eyes
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
fl = FaceLandmarks()

# Loading models
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_img(img, face_detection, hands_detection, blur_enabled=True):
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    face_regions = []  
    if out.multi_face_landmarks is not None:
        # Get the facial landmarks
        landmarks = fl.get_facial_landmarks(img)
        if landmarks is not None:
            convexhull = cv2.convexHull(landmarks)

        for facial_landmarks in out.multi_face_landmarks:
            # Get the bounding box of the face
            bbox = [min([landmark.x for landmark in facial_landmarks.landmark]),
                    min([landmark.y for landmark in facial_landmarks.landmark]),
                    max([landmark.x for landmark in facial_landmarks.landmark]),
                    max([landmark.y for landmark in facial_landmarks.landmark])]

            x1, y1, x2, y2 = [int(val * W) if idx % 2 == 0 else int(val * H) for idx, val in enumerate(bbox)]
            w, h = x2 - x1, y2 - y1

            if w > 0 and h > 0:
                face_regions.append((x1, y1, w, h))  

                face_region = img[y1:y1 + h, x1:x1 + w, :]
                if not face_region.size == 0:
                    if blur_enabled:
                        img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(face_region, (30, 30))
                    else:
                        img[y1:y1 + h, x1:x1 + w, :] = unblur_face(face_region, face_detection)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detection.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_x, hand_y = hand_landmarks.landmark[0].x * W, hand_landmarks.landmark[0].y * H
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

    return img, blur_enabled

def unblur_face(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.multi_face_landmarks is not None:
        for facial_landmarks in out.multi_face_landmarks:
            bbox = [min([landmark.x for landmark in facial_landmarks.landmark]),
                    min([landmark.y for landmark in facial_landmarks.landmark]),
                    max([landmark.x for landmark in facial_landmarks.landmark]),
                    max([landmark.y for landmark in facial_landmarks.landmark])]

            x1, y1, x2, y2 = [int(val * W) if idx % 2 == 0 else int(val * H) for idx, val in enumerate(bbox)]
            w, h = x2 - x1, y2 - y1

            if w > 0 and h > 0:
                face_region = img[y1:y1 + h, x1:x1 + w, :]
                if not face_region.size == 0:
                    img[y1:y1 + h, x1:x1 + w, :] = face_region

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

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mp_face_detection = mp.solutions.face_detection
with face_mesh as face_detection:
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands_detection:

        args = {"mode": 'webcam', "filePath": None}

        if args["mode"] in ["image"]:
            img = cv2.imread(args["filePath"])

            img = process_img(img, face_detection, hands_detection)

            cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

        elif args["mode"] in ['video']:
            cap = cv2.VideoCapture(args["filePath"])
            ret, frame = cap.read()

            output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                           cv2.VideoWriter_fourcc(*'MP4V'),
                                           25,
                                           (frame.shape[1], frame.shape[0]))

            while ret:
                frame = process_img(frame, face_detection, hands_detection)

                output_video.write(frame)

                ret, frame = cap.read()

            cap.release()
            output_video.release()

        elif args["mode"] in ['webcam']:
            cap = cv2.VideoCapture(0)

            ret, frame = cap.read()
            blur_enabled = True  
            while ret:
                frame, blur_enabled = process_img(frame, face_detection, hands_detection, blur_enabled)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  
                    break

                ret, frame = cap.read()

            cap.release()
