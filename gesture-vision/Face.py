import os
import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceLandmarks


mp_drawing = mp.solutions.drawing_utils
fl = FaceLandmarks()

class image_processor:
    def calculate_finger_states(self, hand_landmarks):
        landmarks = np.array([(landmark.x, landmark.y) for landmark in hand_landmarks.landmark])
        finger_states = np.zeros(5, dtype=int)
        
        # Thumb
        finger_states[0] = int(landmarks[2, 0] > landmarks[4, 0])
        
        # Other fingers
        finger_states[1:] = (landmarks[[6, 10, 14, 18], 1] > landmarks[[8, 12, 16, 20], 1]).astype(int)
        
        fingers_count = np.sum(finger_states)

        return finger_states, fingers_count

    def blur_face(self, frame, face_detection):
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
                frame_copy = cv2.blur(frame_copy, (70, 70))
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
    def process_image_with_detection(self, img, hands_detection, blur_enabled=True):
        H, W, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out1 = hands_detection.process(img_rgb)

        hand_regions = []
        if out1.multi_hand_landmarks:
            for hand_landmarks in out1.multi_hand_landmarks:
                hand_x, hand_y = hand_landmarks.landmark[0].x * W, hand_landmarks.landmark[0].y * H

                finger_states, fingers_count = self.calculate_finger_states(hand_landmarks)

                if fingers_count == 1 and finger_states[1] == 1:
                    blur_enabled = True
                else:
                    blur_enabled = False

                hand_regions.append((hand_x, hand_y, finger_states, fingers_count))

        for hand_x, hand_y, finger_states, fingers_count in hand_regions:
            if blur_enabled:
                # Збільшуємо область розмиття вище руки
                blur_radius = 100
                img[int(hand_y) - blur_radius * 2:int(hand_y) + 20,
                int(hand_x) - blur_radius:int(hand_x) + blur_radius] = cv2.blur(
                    img[int(hand_y) - blur_radius * 2:int(hand_y) + 20,
                    int(hand_x) - blur_radius:int(hand_x) + blur_radius], (70, 70))

        return img, blur_enabled
    def process_img(self, img, face_detection, hands_detection, blur_enabled=True):
        H, W, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if blur_enabled and face_detection is not None:
            img = self.blur_face(img, face_detection)
        if hands_detection is not None:
            results = hands_detection.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # hand_x, hand_y = hand_landmarks.landmark[0].x * W, hand_landmarks.landmark[0].y * H
                    finger_states, fingers_count = self.calculate_finger_states(hand_landmarks)

                    if fingers_count == 1 and finger_states[4] == 1:
                        blur_enabled = True
                    elif fingers_count == 2 and finger_states[1] == 1 and finger_states[2] == 1:
                        blur_enabled = False
                    elif fingers_count == 1 and finger_states[1] == 1:
                        self.process_image_with_detection(img, hands_detection)
        return img, blur_enabled

def check_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_detection:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands_detection:
            processor = image_processor()
            args = {"mode": 'webcam', "filePath": None}
            output_dir = './output'
            if args["mode"] in ["image"]:
                img = cv2.imread(args["filePath"])
                check_output_dir(output_dir)
                img = processor.process_img(img, face_detection, hands_detection)

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
                    frame = processor.process_img(frame, face_detection, hands_detection)

                    output_video.write(frame)

                    ret, frame = cap.read()

                cap.release()
                output_video.release()

            elif args["mode"] in ['webcam']:
                cap = cv2.VideoCapture(0)

                ret, frame = cap.read()
                blur_enabled = True
                while ret:
                    frame, blur_enabled = processor.process_img(frame, face_detection, hands_detection, blur_enabled)

                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    ret, frame = cap.read()

                cap.release()

if __name__ == "__main__":
    main()

