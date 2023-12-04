import os
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import font as tkFont
import numpy as np
import time
import threading
import pandas as pd

from tensorflow.keras.models import model_from_json, load_model

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, metrics
from tensorflow.keras.applications import resnet

import threading

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

face_detected = False
current_frame = None
FRAME_INTERVAL_MS = 20
last_real_face_time = None
latest_face_coords = None
taking_attendance = False
registered_username = None

MODEL_DIR = 'models/siamese_model-final'  
VERIF_IMGS_DIR = 'registered_images'
INPUT_IMG_DIR = 'input_images/face.png'
THRESHOLD = 0.2
VERIFICATION_THRESHOLD = 0.3
# Define the Excel file to store attendance
ATTENDANCE_FILE = 'attendance.csv'
import csv
from datetime import datetime

current_page = "main"
name_entry = None
instruction_text = None
confirm_attendance_btn = None

attendance_confirmed = False
countdown_start_time = None
countdown_duration = 5

def load_attendance():
    with open(ATTENDANCE_FILE, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        attendance_data = list(reader)
    return attendance_data

def get_last_action(username, attendance_data):
    for entry in reversed(attendance_data):
        if entry['Employee ID'] == username:
            return entry['Clock-In/Clock-Out Status']
    return None

def record_attendance(username, facial_expression):
    attendance_data = load_attendance()

    last_action = get_last_action(username, attendance_data)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    if last_action == 'Clock-Out' or last_action is None:
        new_status = 'Clock-In'
    else:
        new_status = 'Clock-Out'

    with open(ATTENDANCE_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([current_time, username, new_status, facial_expression])



def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = resnet.ResNet50(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        #pooling='avg',
    )
    
    for i in range(len(pretrained_model.layers)-27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model

def extract_encoder(model):
    # global loaded_model
    encoder = get_encoder((224, 224, 3))
    # encoder = loaded_model
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder

def read_image_pairs(image_pairs):
    pair_images = []
    for pair in image_pairs:  # Read only the first 100 pairs
        image1_path, image2_path = pair
        image1 = cv2.imread(image1_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = cv2.resize(image1, (224, 224))

        image2 = cv2.imread(image2_path)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = cv2.resize(image2, (224, 224))

        image1 = preprocess_input(image1) 
        image2 = preprocess_input(image2) 

        pair_images.append((image1, image2))
    return pair_images

def compute_cosine_similarity(encoder, image_pairs):
    similarities = []

    for pair in image_pairs:
        img1 = np.expand_dims(pair[0], axis=0)
        img2 = np.expand_dims(pair[1], axis=0)

        encoding1 = encoder.predict(img1)
        encoding2 = encoder.predict(img2)

        similarity = cosine_similarity(encoding1, encoding2)[0][0]
        similarities.append(similarity)

    return similarities
        
def load_images(full_path):
    image_pairs = []

    images= os.listdir(full_path)
    for image in images:
        image_path = os.path.join(full_path, image)
        image_pairs.append((INPUT_IMG_DIR, image_path))

    return image_pairs

def verify_user(encoder, callback):
    ENTRIES = os.listdir(VERIF_IMGS_DIR)
    users = []
    
    for entry in ENTRIES:
        current_user = entry

        full_path = os.path.join(VERIF_IMGS_DIR, entry)
        # eg, registered_images\voonTao
        
        img_pairs = read_image_pairs(load_images(full_path))

        similarities = compute_cosine_similarity(encoder, img_pairs)
        print(similarities)

        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        min_similarity = min(similarities)

        user = {
            'username': current_user,
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'min_similarity': min_similarity,
            'total_match': f"{sum(sim > THRESHOLD for sim in similarities)}/{len(img_pairs)}",
        }

        users.append(user)

    print(users)
    # possible_user = max(users, key=lambda user: user['similarity'])

    # Filter and sort users based on avg_similarity and std_similarity
    verified_users = [user for user in users if user['min_similarity'] > THRESHOLD]

    # Users with higher average similarity scores will be considered "greater" in the sorting order.
    # lower standard deviation is preferred because it indicates more consistent similarity scores across all image pairs.
    # lower standard deviations are considered "greater" in the sorting order.
    verified_users.sort(key=lambda u: (u['avg_similarity'], -u['std_similarity']), reverse=True)

    # A user is verified if their average similarity is above 0.5 (VERIFICATION_THRESHOLD)
    # ensuring that the overall similarity level is high.
    if verified_users:
        top_user = verified_users[0]
        if callback:
            if top_user['avg_similarity'] > VERIFICATION_THRESHOLD:
                print('verified')

                detected_emotion = detect_emotion()
                callback(top_user['username'], top_user['avg_similarity'], detected_emotion)
            else:
                print('not verified')
                callback(None, None, None)
    else:
        print('not verified')
        callback(None, None, None)


def detect_emotion():
    # Perform emotion detection on the captured image
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    captured_image_path = INPUT_IMG_DIR

    # Load and preprocess the captured image
    captured_image = cv2.imread(captured_image_path)
    captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)
    captured_image = cv2.resize(captured_image, (48, 48))
    captured_image = np.reshape(captured_image, (1, 48, 48, 1))

    # Predict emotion
    emotion_prediction = EMOTION_MODEL(captured_image)
    detected_emotion_index = np.argmax(emotion_prediction)
    detected_emotion = emotion_labels[detected_emotion_index]

    return detected_emotion


SIAMESE_MODEL = tf.keras.models.load_model(MODEL_DIR)
ENCODER = extract_encoder(SIAMESE_MODEL)

# fps = 1000 ms / 20 ms (frame interval set) = 50 frames per second
# seconds = 250 (num of frames) / 50 (fps) = 5 seconds

# face detection model
FACE_CASCADE_MODEL = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

ANTI_SPOOFING_MODEL = tf.keras.models.load_model('models/antispoofing.h5')

# Emotion model
EMOTION_MODEL_PATH = 'models/facialemotionmodel.h5'
EMOTION_MODEL = load_model(EMOTION_MODEL_PATH)

def set_instruction_text(text):
    global instruction_text
    instruction_text.config(text=text)

def capture_image():
    global latest_face_coords
    
    ret, frame = cap.read()
    
    x, y, w, h = latest_face_coords

    if ret:
        cropped_frame = frame[y:y+h, x:x+w]

        directory = f'input_images'

        if not os.path.exists(directory):
            os.makedirs(directory)

        image_path = f'{directory}/face.png'
        cv2.imwrite(image_path, cropped_frame)

        cv2.waitKey(100)

def capture_and_save_images(user_name, num_images=3):
    global latest_face_coords

    if not cap.isOpened() or latest_face_coords is None:
        return
    
    x, y, w, h = latest_face_coords

    for i in range(num_images):
        ret, frame = cap.read()
        
        if ret:
            cropped_frame = frame[y:y+h, x:x+w]

            # Create directory for user if it doesn't exist
            directory = f'registered_images/{user_name}'
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save image
            image_path = f'{directory}/image_{i+1}.png'
            cv2.imwrite(image_path, cropped_frame)

            # Optionally, display each captured image in the GUI (this can be slow)
            # display_captured_image(image_path)

            # Wait a bit between captures (for example, 100 ms)
            cv2.waitKey(100)

def update_frame():
    global current_frame, face_detected, latest_face_coords, countdown_start_time, attendance_confirmed

    face_detected = False

    # Capture a frame from the camera
    
    if cap.isOpened():
        ret, frame = cap.read()
        current_frame = frame
    else:
        ret = False

    if ret:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE_MODEL.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = current_frame[y-5:y+h+5, x-5:x+w+5]

                img = cv2.resize(face, (160, 160))
                img = preprocess_input(img)
                img = np.expand_dims(img, axis=0)
                prediction = ANTI_SPOOFING_MODEL.predict(img, verbose=0)[0][0]

                if prediction >= 0.9:
                    label = 'Spoof'
                    color = (0, 0, 255)  # Red for spoof
                    face_detected = False
                else:
                    label = 'Real'
                    color = (0, 255, 0)  # Green for real face
                    face_detected = True

                print(prediction, label)

                cv2.putText(current_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), color, 2)
                latest_face_coords = (x, y, w, h)

            cv2image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            label_video.imgtk = imgtk
            label_video.configure(image=imgtk)

    if current_page in ["main"]:
        set_instruction_text(f"Welcome to the attendance system.")
    elif current_page in ["register"]:
        pass
    elif current_page in ["attendance"]:
        if not taking_attendance:
            if face_detected:
                if countdown_start_time is None:
                    countdown_start_time = time.time()
                
                elapsed_time = time.time() - countdown_start_time
                remaining_time = max(0, countdown_duration - int(elapsed_time))
                set_instruction_text(f"Please hold for {remaining_time} more seconds.")

                if remaining_time <= 0:
                    set_instruction_text("Face verified. Please confirm your attendance.")
                    show_attendance_confirmation_button()
            else:
                countdown_start_time = None
                hide_attendance_confirmation_button()
                set_instruction_text("No face detected or fake face.")

        pass
        
    label_video.after(FRAME_INTERVAL_MS, update_frame)
   
def close_camera():
    if cap.isOpened():
        cap.release()

def update_verification_result(username, similarity, detected_emotion):
    # This function updates the GUI with the verification result
    if username is not None:
        set_instruction_text(f"You are verified! Welcome, {username}\n Emotion: {detected_emotion}")
        
        # Call the function to take attendance and record it
        record_attendance(username, detected_emotion)
        display_user_attendance(username)
        # root.after(3000, lambda: switch_to_page("main"))
    else:
        set_instruction_text("Not verified. Please register.")
        # show_register_ui()

def verify_user_thread():
    # This function will be executed in a separate thread
    verify_user(ENCODER, update_verification_result)

def on_take_attendance():
    global taking_attendance

    capture_image()
    close_camera()
    display_captured_image()

    taking_attendance = True

    # Show loading message on the GUI
    set_instruction_text("Processing... Please wait.")
    hide_attendance_confirmation_button()

    # Start the verification process in a separate thread
    threading.Thread(target=verify_user_thread).start()

def on_register():
    global face_detected, taking_attendance, registered_username, name_entry
    
    entered_name = name_entry.get()

    # reopen_camera()
    registered_username = entered_name


    capture_and_save_images(entered_name)
    set_instruction_text(f"Images Saved! Please back and take attendance.")



def reopen_camera():
    global cap
    cap = cv2.VideoCapture(0)

def show_loading_message():
    set_instruction_text("Processing... Please wait.")

def display_captured_image():
    global current_frame

    static_image_path = 'input_images/face.png'

    static_img = Image.open(static_image_path)

    frame_width, frame_height = 600, 500
    static_img_resized = static_img.resize((frame_width, frame_height))

    imgtk = ImageTk.PhotoImage(image=static_img_resized)
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

def show_register_page():
    global current_page, name_entry
    set_instruction_text(f"Please key in your name and click register.")
    clear_frame(frame_right)
    # Add widgets specific to the register page
    register_label = tk.Label(frame_right, text="Register", font=title_font, bg=bg_color, fg=text_color)
    register_label.pack(pady=10)
    
    name_entry = tk.Entry(frame_right, font=entry_font, bg="#ffffff", borderwidth=2)
    name_entry.pack(pady=10, padx=20, fill='x')

    register_btn = tk.Button(frame_right, text="Register", command=on_register, font=button_font, bg=button_color, fg="#ffffff")
    register_btn.pack(pady=10, padx=20, fill='x')
    back_btn = tk.Button(frame_right, text="Back to Main", command=lambda: switch_to_page("main"), font=button_font, bg=button_color, fg="white")
    back_btn.pack(pady=10)
    current_page = "register"

def show_attendance_confirmation_button():
    global confirm_attendance_btn
    confirm_attendance_btn.pack(pady=10)

def hide_attendance_confirmation_button():
    global confirm_attendance_btn
    confirm_attendance_btn.pack_forget()

def switch_to_page(page_name):
    global taking_attendance
    taking_attendance = False

    reopen_camera()

    if page_name == "main":
        show_main_page()
    elif page_name == "register":
        show_register_page()
    elif page_name == "attendance":
        show_attendance_page()

def clear_frame(frame):
    for widget in frame.winfo_children():
        if widget != instruction_text:  # Keep the instruction_text label
            widget.destroy()
        # widget.destroy()

def show_main_page():
    global current_page
    clear_frame(frame_right)
    # Add widgets specific to the main page
    main_page_title = tk.Label(frame_right, text="Main Screen", font=title_font, bg=bg_color, fg=text_color)
    main_page_title.pack(pady=10)
    take_attendance_btn = tk.Button(frame_right, text="Take Attendance", command=lambda: switch_to_page("attendance"), font=button_font, bg=button_color, fg="white")
    take_attendance_btn.pack(pady=10)
    register_btn = tk.Button(frame_right, text="Go to Register", command=lambda: switch_to_page("register"), font=button_font, bg=button_color, fg="white")
    register_btn.pack(pady=10)
    current_page = "main"

def show_attendance_page():
    global current_page, confirm_attendance_btn
    clear_frame(frame_right)
    # Set up the UI elements for the attendance page
    attendance_title = tk.Label(frame_right, text="Attendance", font=title_font, bg=bg_color, fg=text_color)
    attendance_title.pack(pady=10)
    
    confirm_attendance_btn = tk.Button(frame_right, text="Confirm Attendance", command=on_take_attendance, font=button_font, bg=button_color, fg="white")

    back_btn = tk.Button(frame_right, text="Back to Main", command=lambda: switch_to_page("main"), font=button_font, bg=button_color, fg="white")
    back_btn.pack(pady=10)

    current_page = "attendance"
    set_instruction_text("Please position your face within the frame.")

def display_user_attendance(username):
    # Read attendance data
    attendance_data = pd.read_csv(ATTENDANCE_FILE)

    # Filter data for the specific user
    user_data = attendance_data[attendance_data['Employee ID'] == username]

    # Clear existing widgets in the frame
    clear_frame(frame_right)

    # Add a title for the attendance display
    attendance_display_title = tk.Label(frame_right, text=f"Attendance Record for {username}", font=title_font, bg=bg_color, fg=text_color)
    attendance_display_title.pack(pady=10)

    # Create a treeview widget as a table
    columns = ("Date and Time", "Clock-In/Clock-Out Status", "Facial Expression")
    attendance_table = ttk.Treeview(frame_right, columns=columns, show="headings")
    
    # Define the columns
    for col in columns:
        attendance_table.heading(col, text=col)
        attendance_table.column(col, anchor="center", width=100, stretch=tk.YES)

    # Inserting the filtered data into the treeview
    for _, row in user_data.iterrows():
        attendance_table.insert("", tk.END, values=(row['Date and Time'], row['Clock-In/Clock-Out Status'], row['Facial Expression']))

    # Add scrollbars
    vertical_scrollbar = ttk.Scrollbar(frame_right, orient="vertical", command=attendance_table.yview)
    vertical_scrollbar.pack(side="right", fill="y")
    attendance_table.configure(yscrollcommand=vertical_scrollbar.set)

    horizontal_scrollbar = ttk.Scrollbar(frame_right, orient="horizontal", command=attendance_table.xview)
    horizontal_scrollbar.pack(side="bottom", fill="x")
    attendance_table.configure(xscrollcommand=horizontal_scrollbar.set)

    # Pack the table with controlled expansion
    attendance_table.pack(expand=False, fill="both", pady=10)

    # Add a back button to return to the main page
    back_btn = tk.Button(frame_right, text="Back to Main", command=lambda: switch_to_page("main"), font=button_font, bg=button_color, fg="white")
    back_btn.pack(pady=10)

# Initialize the main window
root = tk.Tk()
root.geometry("1200x500")
root.title("Face Recognition Attendance System with Liveness Detection")

# Define fonts and styles
title_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
button_font = entry_font = tkFont.Font(family="Helvetica", size=12)
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

cap = cv2.VideoCapture(0)

instruction_text = tk.Label(frame_right, font=instruction_font, bg=bg_color, fg=text_color)
instruction_text.pack(pady=10)

show_main_page()

update_frame()

root.mainloop()

close_camera()
