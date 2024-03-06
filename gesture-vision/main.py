import cv2
import mediapipe as mp
import customtkinter
import tkinter as tk
from Face import image_processor
from tkinter import Text, Scrollbar
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

        # Initialize image processor
        self.img_proc = image_processor()

    # Functions creating

    # CAMERA

    def toggle_camera(self):
        self.is_camera_enabled = not self.is_camera_enabled
        if self.is_camera_enabled:
            self.toggle_camera_button.configure(fg_color="#C850C0", hover_color="#c85090")
            self.start_camera()
            if self.is_debug_enabled:
                self.send_debug_message("camera enabled")
        else:
            self.toggle_camera_button.configure(fg_color="#1f538d", hover_color="#14375e")
            self.stop_camera()
            if self.is_debug_enabled:
                self.send_debug_message("camera disabled")

    def start_camera(self):
        self.video = cv2.VideoCapture(0)
        self.timer = self.after(10, self.update_frame)

    def stop_camera(self):
        if self.timer:
            self.after_cancel(self.timer)
        self.video.release()
        self.image_label.config(image=self.gray_photo)  # При отключении камеры отображаем серый прямоугольник

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            frame = self.img_proc.process_start(frame, self.is_handblur_enabled, self.is_gesture_enabled)
            # Зеркально отразить кадр по горизонтали
            #frame = cv2.flip(frame, 1)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            img = ImageTk.PhotoImage(image=img)

            self.image_label.img = img
            self.image_label.config(image=img)

        if self.is_camera_enabled:
            self.timer = self.after(10, self.update_frame)

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

if __name__ == "__main__":
    app = FaceBlurApp()
    app.mainloop()