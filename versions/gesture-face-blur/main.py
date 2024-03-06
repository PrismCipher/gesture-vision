import os
import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceLandmarks

# Initialize objects to detect hands and face
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
fl = FaceLandmarks()

def process_img(img, face_detection, hands_detection, blur_enabled=True):
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if blur_enabled:
        img = blur_face(img, face_detection)

    results = hands_detection.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #hand_x, hand_y = hand_landmarks.landmark[0].x * W, hand_landmarks.landmark[0].y * H
            finger_states, fingers_count = calculate_finger_states(hand_landmarks)

            if fingers_count == 1:
                blur_enabled = True
            elif fingers_count == 5:
                blur_enabled = False

    return img, blur_enabled

def blur_face(frame, face_detection):
    frame_copy = frame.copy()
    height, width, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    # Face landmark detection
    if out.multi_face_landmarks is not None:
        # Get the facial landmarks
        landmarks = fl.get_facial_landmarks(frame)
        if landmarks is not None and landmarks.ndim == 2 and landmarks.shape[1] == 2:
            convexhull = cv2.convexHull(landmarks)
            # Face blurrying
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, convexhull, 255)

            # Extract the face
            frame_copy = cv2.blur(frame_copy, (30, 30))
            face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

            # Extract background
            background_mask = cv2.bitwise_not(mask)
            background = cv2.bitwise_and(frame, frame, mask=background_mask)

            # Final result
            result = cv2.add(background, face_extracted)
        else:
            result = frame
    else:
        result = frame
    return result

def calculate_finger_states(hand_landmarks):
    finger_states = []
    finger_tip_landmarks = [4, 8, 12, 16, 20]
    finger_base_landmarks = [3, 6, 10, 14, 18]

    for tip, base in zip(finger_tip_landmarks, finger_base_landmarks):
        if hand_landmarks.landmark[base].y > hand_landmarks.landmark[tip].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    fingers_count = sum(finger_states)

    return finger_states, fingers_count

def check_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    # Loading models
    with mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_detection:
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands_detection:

            args = {"mode": 'webcam', "filePath": None}
            output_dir = './output'
            if args["mode"] in ["image"]:
                img = cv2.imread(args["filePath"])

                img = process_img(img, face_detection, hands_detection)
                check_output_dir(output_dir)
                cv2.imwrite(os.path.join(output_dir, 'output.png'), img)

            elif args["mode"] in ['video']:
                cap = cv2.VideoCapture(args["filePath"])
                ret, frame = cap.read()
                check_output_dir(output_dir)
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

                cv2.destroyAllWindows()
                cap.release()

if __name__ == "__main__":
    main()
