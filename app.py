import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import font as tkFont
import numpy as np
import time

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

frame_counter = 0
face_detected = False
button_shown = False
current_frame = None
frame_interval_ms = 20
last_real_face_time = None

# fps = 1000 ms / 20 ms (frame interval set) = 50 frames per second
# seconds = 250 (num of frames) / 50 (fps) = 5 seconds

# face detection model
face_cascade = cv2.CascadeClassifier("models/face_detection.xml")

# anti-spoofing model
json_file = open('models/anti_spoofing.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('models/anti_spoofing.h5')

def has_elapsed_seconds(target_seconds):
    global frame_counter, frame_interval_ms
    
    milliseconds_per_second = 1000
    elapsed_seconds = (frame_counter * frame_interval_ms) / milliseconds_per_second
    
    return elapsed_seconds >= target_seconds

def set_instruction_text(text):
    instruction_text.config(text=text)

def is_user_verified():
    # TODO: face recognition model
    return False

def capture_and_save_images(user_name, num_images=30):
    # TODO: capture images from camera register new face id

    pass
    # if not cap.isOpened():
    #     return
    
    # for i in range(num_images):
    #     ret, frame = cap.read()
    #     if ret:
    #         # Create directory for user if it doesn't exist
    #         directory = f'input_images/{user_name}'
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)

    #         # Save image
    #         image_path = f'{directory}/image_{i+1}.png'
    #         cv2.imwrite(image_path, frame)

    #         # Optionally, display each captured image in the GUI (this can be slow)
    #         # display_captured_image(image_path)

    #         # Wait a bit between captures (for example, 100 ms)
    #         cv2.waitKey(100)


def update_frame():
    global frame_counter, button_shown, current_frame, face_detected

    face_detected = False

    # Capture a frame from the camera
    if cap.isOpened():
        ret, frame = cap.read()
        current_frame = frame

    if ret:
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = current_frame[y-5:y+h+5, x-5:x+w+5]
            resized_face = cv2.resize(face, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)
            preds = model.predict(resized_face, verbose=0)[0]
            
            if preds <= 0.1:
                label = 'Real'
                color = (0, 255, 0)  # Green for real face
                face_detected = True
            else:
                label = 'Spoof'
                color = (0, 0, 255)  # Red for spoof
                face_detected = False

            cv2.putText(current_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), color, 2)

        cv2image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)

    # Check for face detection
    if face_detected:
        frame_counter += 1
    else:
        frame_counter = 0
        button_shown = False

    print(f"face_detected: {face_detected}")
    # Check if x seconds have passed since face detection
    if has_elapsed_seconds(2) and not button_shown:
        print("Show button")
        show_button()
        set_instruction_text("Please click the button to take attendance.")
        button_shown = True
    
    if not face_detected:
        print("Hide button")
        hide_button()

    label_video.after(frame_interval_ms, update_frame)
        

def show_button():
    take_attendance_btn.pack(pady=10)

def hide_button():
    take_attendance_btn.pack_forget()

def close_camera():
    if cap.isOpened():
        cap.release()

def on_take_attendance():
    global button_shown

    close_camera()

    # TODO: store image into directory

    display_captured_image()

    if is_user_verified():
        set_instruction_text("You are verified! Welcome, [Username]")
        hide_button()
        # show_button()
    else:
        reopen_camera()
        set_instruction_text("Not verified. Please register.")
        show_register_ui()
        

def show_register_ui():
    global name_entry, register_btn
    hide_button()

    # Styling for entry and button
    entry_font = tkFont.Font(family="Helvetica", size=12)
    button_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
    entry_bg_color = "#ffffff"
    button_color = "#4a7a8c"
    button_text_color = "#ffffff"

    # Entry for name input
    name_entry = tk.Entry(frame_right, font=entry_font, bg=entry_bg_color, borderwidth=2)
    name_entry.pack(pady=10, padx=20, fill='x')  # Fill x-direction for wider entry

    # Register button
    register_btn = tk.Button(frame_right, text="Register", command=on_register, font=button_font, bg=button_color, fg=button_text_color)
    register_btn.pack(pady=10, padx=20, fill='x')  # Fill x-direction for wider button


def on_register():
    global button_shown, frame_counter
    entered_name = name_entry.get()
    

    set_instruction_text(f"Registered as {entered_name}. Stay in front of the camera.")
    name_entry.pack_forget()
    register_btn.pack_forget()

    show_loading_message()

    reopen_camera()

    capture_and_save_images(entered_name)

    # Prompt user to stay in front of the camera for re-verification
    set_instruction_text("Please stay in front of the camera for verification again.")
    button_shown = False
    frame_counter = 0


def reopen_camera():
    global cap
    cap = cv2.VideoCapture(0)

def show_loading_message():
    set_instruction_text("Processing... Please wait.")

def display_captured_image():
    global current_frame

    static_image_path = 'input_images/face1.png'

    # TODO: replace with something else
    # if current_frame is not None:
    #     cv2.imwrite(static_image_path, cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))

    static_img = Image.open(static_image_path)

    frame_width, frame_height = 600, 500
    static_img_resized = static_img.resize((frame_width, frame_height))

    imgtk = ImageTk.PhotoImage(image=static_img_resized)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

# Initialize the main window
root = tk.Tk()
root.title("Face Recognition Attendance System with Liveness Detection")

# Define fonts and styles
title_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
button_font = tkFont.Font(family="Helvetica", size=12)
instruction_font = tkFont.Font(family="Helvetica", size=14)

# Define colors
bg_color = "#f0f0f0"
button_color = "#4a7a8c"
text_color = "#333333"

# Set window background color
root.configure(bg=bg_color)

# Create frames for layout
frame_left = tk.Frame(root, width=600, height=500)
frame_left.pack(side="left", fill="both", expand=True)

frame_right = tk.Frame(root, width=200, height=500)
frame_right.pack(side="right", fill="both", expand=True)

# Add a label for the video frame in the left frame
label_video = tk.Label(frame_left, bg=bg_color)
label_video.pack(padx=10, pady=10)

# Add UI elements in the right frame
label_text = tk.Label(frame_right, text="Face Recognition Attendance System\nwith Liveness Detection", font=title_font, bg=bg_color, fg=text_color)
label_text.pack(pady=10)

instruction_text = tk.Label(frame_right, font=instruction_font, bg=bg_color, fg=text_color, text="Please stay in front of the camera for verification.")
instruction_text.pack(pady=10)

# Create the button but do not pack it yet
take_attendance_btn = tk.Button(frame_right, text="Take Attendance", command=on_take_attendance, font=button_font, bg=button_color, fg="white")

# Initialize camera
cap = cv2.VideoCapture(0)

# Start the update process for the camera
update_frame()

# Start the GUI
root.mainloop()

# Release the camera when the window is closed
close_camera()