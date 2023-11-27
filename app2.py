import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, metrics
from tensorflow.keras.applications import resnet

MODEL_DIR = 'models/siamese_model-final'  
VERIF_IMGS_DIR = 'registered_images'
THRESHOLD = 0.2
VERIFICATION_THRESHOLD = 0.3
CROPPED_FACE_SIZE = (128, 128)  # Desired size for cropped faces
INPUT_IMG_DIR = 'images/jia.jpeg'
scale_factor = 0.5

# Load the model
siamese_model = tf.keras.models.load_model(MODEL_DIR)

def get_encoder(input_shape):
    """ Returns the image encoding model """
    pretrained_model = resnet.ResNet50(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
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
    encoder = get_encoder((224, 224, 3))
    i = 0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i += 1
    return encoder

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
    images = os.listdir(full_path)
    for image in images:
        image_path = os.path.join(full_path, image)
        image_pairs.append((INPUT_IMG_DIR, image_path))
    return image_pairs

def read_image_pairs(image_pairs):
    pair_images = []
    for pair in image_pairs:
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


def verify_user(encoder, face_image):
    ENTRIES = os.listdir(VERIF_IMGS_DIR)
    users = []
    for entry in ENTRIES:
        current_user = entry
        full_path = os.path.join(VERIF_IMGS_DIR, entry)
        img_pairs = read_image_pairs(load_images(full_path))
        similarities = compute_cosine_similarity(encoder, img_pairs)
        print("Similarity: ", similarities)
        predictions = [1 if sim > THRESHOLD else 0 for sim in similarities]
        print("predictions: ", predictions)
        similar_pairs_count = sum(predictions)
        print("similar_pairs_count: ", similar_pairs_count)
        similarity = similar_pairs_count / len(img_pairs) if img_pairs else 0
        user = {
            'username': current_user,
            'similarity': similarity
        }
        users.append(user)
    possible_user = max(users, key=lambda user: user['similarity'])
    username = possible_user['username']
    similarity = possible_user['similarity']
    if similarity > VERIFICATION_THRESHOLD:
        return username, similarity
    else:
        return None, None

def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_regions = []

    for (x, y, w, h) in faces:
        face_region = img[y:y+h, x:x+w]
        face_regions.append((x, y, w, h, face_region))

    return face_regions


face_regions = detect_faces(INPUT_IMG_DIR)
encoder = extract_encoder(siamese_model)
original_image = cv2.imread(INPUT_IMG_DIR)

# Resize the original image
original_image = cv2.resize(original_image, None, fx=scale_factor, fy=scale_factor)

for (x, y, w, h, face_image) in face_regions:
    # Resize the cropped face to the desired size
    resized_face = cv2.resize(face_image, CROPPED_FACE_SIZE)
    username, similarity = verify_user(encoder, resized_face)
    if username:
        label = f"User: {username}"
        color = (0, 255, 0)  # Green color for recognized users
    else:
        label = "Unknown user"
        color = (0, 0, 255)  # Red color for unknown users

    # Draw bounding box and label on the resized original image
    cv2.rectangle(original_image, (int(x * scale_factor), int(y * scale_factor)), (int((x + w) * scale_factor), int((y + h) * scale_factor)), color, 2)
    cv2.putText(original_image, label, (int(x * scale_factor), int((y - 10) * scale_factor)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

# Display the resized original image with bounding boxes and labels
cv2.imshow("Face Recognition", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()