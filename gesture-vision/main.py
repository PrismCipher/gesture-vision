import os
import cv2
import mediapipe as mp
import numpy as np

def calculate_finger_states(hand_landmarks):
    """
    Обчислення стану пальців на основі ландмарок руки.

    Аргументи:
    - hand_landmarks: Ландмарки руки.

    Повертає:
    - Список, який представляє стан кожного пальця (0 або 1).
    - Загальна кількість піднятих пальців.
    """
    finger_states = [0, 0, 0, 0, 0]
    if hand_landmarks.landmark[2].x > hand_landmarks.landmark[4].x:  # Великий
        finger_states[0] = 1
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[20].x:
        finger_states[0] = hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x

    else:
        finger_states[0] = hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x

    if hand_landmarks.landmark[6].y > hand_landmarks.landmark[8].y:
        finger_states[1] = 1

    if hand_landmarks.landmark[10].y > hand_landmarks.landmark[12].y:  # середній
        finger_states[2] = 1

    if hand_landmarks.landmark[14].y > hand_landmarks.landmark[16].y:
        finger_states[3] = 1

    if hand_landmarks.landmark[18].y > hand_landmarks.landmark[20].y:
        finger_states[4] = 1

    fingers_count = sum(finger_states)

    return finger_states, fingers_count

def unblur_face(img, face_detection):
    """
    Убрать размытие с лиц на изображении.

    Аргументы:
    - img: Входное изображение.
    - face_detection: Модель обнаружения лиц.

    Возвращает:
    - Изображение с не размытыми лицами.
    """
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = int(bbox.xmin * W), int(bbox.ymin * H), int(bbox.width * W), int(bbox.height * H)

            if w > 0 and h > 0:
                face_region = img[y1:y1 + h, x1:x1 + w, :]
                if not face_region.size == 0:
                    img[y1:y1 + h, x1:x1 + w, :] = face_region

    return img
def process_image_with_detection(img, face_detection, hands_detection, blur_enabled=True):
    """
    Обробка зображення шляхом виявлення обличчя та рук, та застосування розмиття на основі жестів рук.

    Аргументи:
    - img: Вхідне зображення.
    - face_detection: Модель виявлення облич.
    - hands_detection: Модель виявлення рук.
    - blur_enabled: Булеве значення, що вказує, чи застосовувати розмиття.

    Повертає:
    - Оброблене зображення.
    - Поточний стан розмиття.
    """

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out1 = hands_detection.process(img_rgb)

    hand_regions = []
    if out1.multi_hand_landmarks:
        for hand_landmarks in out1.multi_hand_landmarks:
            hand_x, hand_y = hand_landmarks.landmark[0].x * W, hand_landmarks.landmark[0].y * H

            finger_states, fingers_count = calculate_finger_states(hand_landmarks)

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
def process_img(img, face_detection, hands_detection, blur_enabled=True):
    """
    Обработка изображения путем обнаружения лиц и рук, и применения размытия на основе жестов рук.

    Аргументы:
    - img: Входное изображение.
    - face_detection: Модель обнаружения лиц.
    - hands_detection: Модель обнаружения рук.
    - blur_enabled: Булево значение, указывающее, применять ли размытие.

    Возвращает:
    - Обработанное изображение.
    - Текущее состояние размытия.
    """
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

                if fingers_count == 1 and finger_states[4]==1:
                    blur_enabled = True
                elif fingers_count == 2 and finger_states[1]==1 and finger_states[2] ==1:
                    blur_enabled = False
                elif fingers_count == 1 and finger_states[1] == 1:
                    process_image_with_detection(img, face_detection, hands_detection, blur_enabled=True)



    return img, blur_enabled

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
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

