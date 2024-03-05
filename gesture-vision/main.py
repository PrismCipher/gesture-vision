import cv2
import mediapipe as mp
import customtkinter
import tkinter as tk
import threading
import queue
from tkinter import Text, Scrollbar
from Face import image_processor
from facial_landmarks import FaceLandmarks
from PIL import Image, ImageTk

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")


class FaceBlurApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure Window

        self.title("FaceBlurApp")

        # Configure Settings

        self.video = cv2.VideoCapture(0)
        self.is_camera_enabled = False
        self.is_debug_enabled = False
        self.is_gesture_enabled = False
        self.is_faceblur_enabled = False
        self.is_handblur_enabled = False
        # creating widgets

        self.image_label = tk.Label(master=self)
        self.image_label.grid(row=0, column=0, columnspan=2, padx=0, pady=(0, 0), sticky="nsew")

        self.button_frame = customtkinter.CTkFrame(master=self, corner_radius=0)
        self.button_frame.grid(row=1, column=0,sticky="nsew")

        self.toggle_gesture_button = customtkinter.CTkButton(self.button_frame, text="Gesture Recognition",
                                                             command=self.toggle_gesture_recognition)
        self.toggle_gesture_button.grid(row=0, column=0, padx=20, pady=15, sticky="ew")

        self.toggle_camera_button = customtkinter.CTkButton(self.button_frame, text="Camera", command=self.toggle_camera)
        self.toggle_camera_button.grid(row=0, column=1, padx=20, pady=15, sticky="ew")

        self.faceblur_button = customtkinter.CTkButton(self.button_frame, text="Face Blur", command=self.toggle_faceblur)
        self.faceblur_button.grid(row=0, column=2, padx=20, pady=15, sticky="ew")

        self.handblur_button = customtkinter.CTkButton(self.button_frame, text="Hand blur", command=self.toggle_handblur)
        self.handblur_button.grid(row=0, column=3, padx=20, pady=15, sticky="ew")

        self.debug_button = customtkinter.CTkButton(self.button_frame, text="Debug", command=self.toggle_debug_console)
        self.debug_button.grid(row=0, column=5, padx=20, pady=15, sticky="ew")

        self.gray_image = Image.new("RGB", (640, 480), "gray")  # Создаем серый прямоугольник
        self.gray_photo = ImageTk.PhotoImage(self.gray_image)
        self.image_label.config(image=self.gray_photo)  # Устанавливаем серый прямоугольник по умолчанию

        self.debug_console = None
        self.timer = None

        self.toplevel_window = None

        self.camera_lock = threading.Lock()

    # Functions creating

    # CAMERA

    def toggle_camera(self):
        self.is_camera_enabled = not self.is_camera_enabled
        if self.is_camera_enabled:
            self.toggle_camera_button.configure(fg_color="#C850C0", hover_color="#c85090")
            self.start_camera()
            self.start_threads()
            if self.is_debug_enabled:
                self.send_debug_message("camera enabled")
        else:
            self.toggle_camera_button.configure(fg_color="#1f538d", hover_color="#14375e")
            self.stop_camera()
            self.stop_threads()
            if self.is_debug_enabled:
                self.send_debug_message("camera disabled")

    def start_camera(self):
        self.video = cv2.VideoCapture(0)
        self.timer = self.after(5, self.update_frame)

    def stop_camera(self):
        if self.timer:
            self.after_cancel(self.timer)
        self.video.release()
        self.image_label.config(image=self.gray_photo)  # При отключении камеры отображаем серый прямоугольник

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            self.transfer_frame(frame)
            if self.queue_process_interface.empty():
                # Зеркально отразить кадр по горизонтали
                frame = cv2.flip(frame, 1)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_image)
                img = ImageTk.PhotoImage(image=img)

                self.image_label.img = img
                self.image_label.config(image=img)
            else:
                frame = self.queue_process_interface.get()
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_image)
                img = ImageTk.PhotoImage(image=img)
                self.image_label.img = img
                self.image_label.config(image=img)

        if self.is_camera_enabled:
            self.timer = self.after(10, self.update_frame)

    # FACE BLUR

    def toggle_faceblur(self):
        self.is_faceblur_enabled = not self.is_faceblur_enabled
        if self.is_faceblur_enabled:
            self.faceblur_button.configure(fg_color="#C850C0", hover_color="#c85090")
            if self.is_debug_enabled:
                self.send_debug_message("Face blur enabled")
        else:
            self.faceblur_button.configure(fg_color="#1f538d", hover_color="#14375e")
            if self.is_debug_enabled:
                self.send_debug_message("Face blur disabled")

    # HANDS BLUR

    def toggle_handblur(self):
        self.is_handblur_enabled = not self.is_handblur_enabled
        if self.is_handblur_enabled:
            self.handblur_button.configure(fg_color="#C850C0", hover_color="#c85090")
            if self.is_debug_enabled:
                self.send_debug_message("Hand blur enabled")
        else:
            self.handblur_button.configure(fg_color="#1f538d", hover_color="#14375e")
            if self.is_debug_enabled:
                self.send_debug_message("Hand blur disabled")

    # DEBUG

    def toggle_debug_console(self):
        self.is_debug_enabled = not self.is_debug_enabled
        if self.is_debug_enabled:
            self.debug_button.configure(fg_color="#C850C0", hover_color="#c85090")
            self.open_debug_console()
        else:
            self.debug_button.configure(fg_color="#1f538d", hover_color="#14375e")
            self.close_debug_console()

    def open_debug_console(self):
        if not self.debug_console:
            self.debug_console = tk.Toplevel()
            self.debug_console.title("Debug Console")
            self.debug_console.resizable(False, False)  # Запрещаем изменение размеров окна

            self.debug_text = Text(self.debug_console)
            self.debug_text.pack(expand=True, fill=tk.BOTH)

            scrollbar = Scrollbar(self.debug_console, command=self.debug_text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.debug_text.config(yscrollcommand=scrollbar.set)

    def close_debug_console(self):
        if self.debug_console:
            self.debug_console.destroy()
            self.debug_console = None
            
    def send_debug_message(self, message):
        if self.is_debug_enabled and self.debug_text:
            self.debug_text.insert(tk.END, message + "\n")
            self.debug_text.see(tk.END)

    # GESTURE RECOGNITION

    def toggle_gesture_recognition(self):
        self.is_gesture_enabled = not self.is_gesture_enabled
        if self.is_gesture_enabled:
            self.toggle_gesture_button.configure(fg_color="#C850C0", hover_color="#c85090")
            if self.is_debug_enabled:
                self.send_debug_message("gesture recognition enabled")
        else:
            self.toggle_gesture_button.configure(fg_color="#1f538d", hover_color="#14375e")
            if self.is_debug_enabled:
                self.send_debug_message("gesture recognition disabled")
    
    # SCARY THING - THREADING

    queue_interface_process = queue.Queue()
    queue_process_interface = queue.Queue()

    def transfer_frame(self, frame):
        self.queue_interface_process.put(frame)

    def start_threads(self):
        self.process_image_thread = threading.Thread(target=self.process_image_thread)
        self.get_result_thread = threading.Thread(target=self.get_result_thread)
        self.process_image_thread.start()
        self.get_result_thread.start()

    def stop_threads(self):
        # Add a way to stop the threads
        self.is_camera_enabled = False
        # Wait for the threads to finish
        self.process_image_thread.join()
        self.get_result_thread.join()

    def process_image_thread(self, frame):
        while self.is_camera_enabled:
            processor = image_processor()
            mp_face_mesh = mp.solutions.face_mesh
            mp_hands = mp.solutions.hands
            face_detection, hands_detection = None, None
            if self.is_faceblur_enabled:
                face_detection = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            if self.is_gesture_enabled:
                hands_detection = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
            while True:
                if not self.queue_interface_process.empty():
                    frame = self.queue_interface_process.get()
                    frame = processor.process_img(frame, face_detection, hands_detection, self.is_faceblur_enabled)
                    self.queue_process_interface.put(frame)
                    
    def get_result_thread(self):
        while self.is_camera_enabled:
            if not self.queue_process_interface.empty():
                frame = self.queue_process_interface.get()
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_image)
                img = ImageTk.PhotoImage(image=img)
                self.image_label.img = img
                self.image_label.config(image=img)


if __name__ == "__main__":
    app = FaceBlurApp()
    app.mainloop()
