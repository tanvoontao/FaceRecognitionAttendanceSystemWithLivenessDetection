import os
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import font as tkFont
import numpy as np
import time

from tensorflow.keras.models import model_from_json

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, metrics
from tensorflow.keras.applications import resnet

import threading

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

face_detected = False
current_frame = None
FRAME_INTERVAL_MS = 20
last_real_face_time = None
latest_face_coords = None
registering = True
taking_attendance = True
registered_username = None

MODEL_DIR = 'models/siamese_model-final'  
VERIF_IMGS_DIR = 'registered_images'
INPUT_IMG_DIR = 'input_images/face.png'
THRESHOLD = 0.5
VERIFICATION_THRESHOLD = 0.5

SIAMESE_MODEL = tf.keras.models.load_model(MODEL_DIR)

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
        # total_images_count += 2 * len(img_pairs)

        similarities = compute_cosine_similarity(encoder, img_pairs)

        # print(similarities)
        predictions = [1 if sim > THRESHOLD else 0 for sim in similarities]

        similar_pairs_count = sum(predictions)

        # similarity = total_match / total_imgs
        similarity = similar_pairs_count / len(img_pairs) if img_pairs else 0

        user = {
            'username': current_user,
            'similarity': similarity
        }

        users.append(user)

    possible_user = max(users, key=lambda user: user['similarity'])

    username = possible_user['username']
    similarity = possible_user['similarity']

    print(users)    
    if callback:
        callback(username, similarity)

    if callback:
        if (similarity > VERIFICATION_THRESHOLD):
            print('verified')
            callback(username, similarity)
        else:
            print('not verified')
            callback(None, None)

    # print(possible_user)
    # if (similarity > VERIFICATION_THRESHOLD):
    #     print('verified')
    #     return username, similarity
    # else:
    #     print('not verified')
    #     return None, None

ENCODER = extract_encoder(SIAMESE_MODEL)



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

def set_instruction_text(text):
    instruction_text.config(text=text)

def is_user_verified():
    # TODO: face recognition model
    username, similarity = verify_user(ENCODER)
    if username is not None:
        return True
    return False

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

def capture_and_save_images(user_name, num_images=30):
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
    global current_frame, face_detected, last_real_face_time, latest_face_coords, registering

    face_detected = False

    # Capture a frame from the camera
    
    if cap.isOpened():
        ret, frame = cap.read()
        current_frame = frame
    else:
        ret = False

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
            latest_face_coords = (x, y, w, h)

        cv2image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        label_video.imgtk = imgtk
        label_video.configure(image=imgtk)

    if taking_attendance:
        if face_detected:
            last_real_face_time = last_real_face_time or time.time()
            elapsed = time.time() - last_real_face_time
            remaining_time = max(0, 2 - int(elapsed))  # 2 seconds countdown

            if elapsed < 2:
                instruction = f"Stay in front of the camera for {remaining_time} more seconds. "
                hide_button()

                if registered_username is not None:
                    instruction += f"\nPlease, {registered_username}!"
                set_instruction_text(instruction)
            else:
                instruction = "Please click the button to take attendance. "
                show_button()
                if registered_username is not None:
                    instruction += f"please, {registered_username}. "
                set_instruction_text(instruction)
        else:
            last_real_face_time = None
            hide_button()
            set_instruction_text("No face detected. Please stay in front of the camera.")

    if registering:
        pass
    
   
    label_video.after(FRAME_INTERVAL_MS, update_frame)
        

def show_button():
    take_attendance_btn.pack(pady=10)

def hide_button():
    take_attendance_btn.pack_forget()

def close_camera():
    if cap.isOpened():
        cap.release()

# def on_take_attendance():
#     global taking_attendance

#     capture_image()

#     close_camera()
#     display_captured_image()

#     username, similarity = verify_user(ENCODER)
#     is_user_verified = False

#     if username is not None:
#         is_user_verified = True
    

#     if is_user_verified:
#         set_instruction_text(f"You are verified! Welcome, {username}")
#         hide_button()
#         taking_attendance = False
        
#     else:
#         reopen_camera()
#         set_instruction_text("Not verified. Please register.")
#         show_register_ui()
#         taking_attendance = False

def update_verification_result(username, similarity):
    # This function updates the GUI with the verification result
    if username is not None:
        set_instruction_text(f"You are verified! Welcome, {username}")
        hide_button()
    else:
        set_instruction_text("Not verified. Please register.")
        show_register_ui()

def verify_user_thread():
    # This function will be executed in a separate thread
    verify_user(ENCODER, update_verification_result)

def on_take_attendance():
    global taking_attendance

    capture_image()
    close_camera()
    display_captured_image()

    # Show loading message on the GUI
    set_instruction_text("Processing... Please wait.")

    # Start the verification process in a separate thread
    threading.Thread(target=verify_user_thread).start()

    # Update the taking_attendance flag
    taking_attendance = False
        

def show_register_ui():
    global name_entry, register_btn, registering
    registering = True

    hide_button()
    
    set_instruction_text(f"Please key in your name and click register.")
    name_entry.pack(pady=10, padx=20, fill='x')
    register_btn.pack(pady=10, padx=20, fill='x')
    


def on_register():
    global registering, face_detected, taking_attendance, registered_username
    
    
    entered_name = name_entry.get()
    
    name_entry.pack_forget()
    register_btn.pack_forget()

    reopen_camera()
    registered_username = entered_name


    capture_and_save_images(entered_name)

    registering = False
    taking_attendance = True


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

# Initialize the main window
root = tk.Tk()
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

# Add UI elements in the right frame
label_text = tk.Label(frame_right, text="Face Recognition Attendance System\nwith Liveness Detection", font=title_font, bg=bg_color, fg=text_color)
label_text.pack(pady=10)

instruction_text = tk.Label(frame_right, font=instruction_font, bg=bg_color, fg=text_color, text="Please stay in front of the camera for verification.")
instruction_text.pack(pady=10)

# ---------- Take Attendance UI ---------- #
# Create the button but do not pack it yet
take_attendance_btn = tk.Button(frame_right, text="Take Attendance", command=on_take_attendance, font=button_font, bg=button_color, fg="white")

# ---------- Register UI ---------- #
# Entry for name input
name_entry = tk.Entry(frame_right, font=entry_font, bg="#ffffff", borderwidth=2)

# Register button
register_btn = tk.Button(frame_right, text="Register", command=on_register, font=button_font, bg=button_color, fg="#ffffff")

# Initialize camera
cap = cv2.VideoCapture(1)

# Start the update process for the camera
update_frame()

# Start the GUI
root.mainloop()

# Release the camera when the window is closed
close_camera()