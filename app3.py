import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import  Sequential
from tensorflow.keras import layers 
from tensorflow.keras.applications import resnet

MODEL_DIR = 'models/siamese_model-final'  
VERIF_IMGS_DIR = 'registered_images'
INPUT_IMG_DIR = 'input_images/cropped_face.png'
ORI_IMG_DIR = 'input_images/ori_face.png'
THRESHOLD = 0.5
VERIFICATION_THRESHOLD = 0.5
DIR = 'input_images'

# Load the model
siamese_model = tf.keras.models.load_model(MODEL_DIR)

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

        # encoding1 = encoder.predict(img1, verbose=0)
        # encoding2 = encoder.predict(img2, verbose=0)

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

def verify_user(encoder):
    ENTRIES = os.listdir(VERIF_IMGS_DIR)
    users = []
    
    for entry in ENTRIES:
        current_user = entry

        full_path = os.path.join(VERIF_IMGS_DIR, entry)
        # eg, registered_images\voonTao
        
        img_pairs = read_image_pairs(load_images(full_path))

        similarities = compute_cosine_similarity(encoder, img_pairs)
        
        print(similarities)
        predictions = [1 if sim > THRESHOLD else 0 for sim in similarities]

        similar_pairs_count = sum(predictions)

        similarity = similar_pairs_count / len(img_pairs) if img_pairs else 0

        user = {
            'username': current_user,
            'similarity': similarity
        }

        users.append(user)

    print(users)
    possible_user = max(users, key=lambda user: user['similarity'])

    username = possible_user['username']
    similarity = possible_user['similarity']

    if (similarity > VERIFICATION_THRESHOLD):
        return username, similarity
    else:
        return None, None

def read_n_crop_face(ori_img, faces):
    for (x, y, w, h) in faces:
        cropped_img = ori_img[y:y+h, x:x+w]
        cv2.imwrite(f'{DIR}/cropped_face.png', cropped_img)
        break

def show_img(ori_img, faces, username, similarity):
    color = GREEN_COLOR if username is not None else RED_COLOR
    # label = f'{username}, {similarity*100:.2f}%' if username is not None else 'Unknown user'
    label = f'{username}' if username is not None else 'Unknown user'

    for (x, y, w, h) in faces:
        cv2.putText(ori_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(ori_img, (x, y), (x + w, y + h), color, 2)
        break

    cv2.imshow('Face Recognition', ori_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# face detection model
FACE_CASCADE_MODEL = cv2.CascadeClassifier("models/face_detection.xml")

GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)


if __name__ == "__main__":
    ori_img = cv2.imread(ORI_IMG_DIR)
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE_MODEL.detectMultiScale(gray, 1.3, 5)

    # Check if any faces are detected
    if len(faces) == 0:
        print("No faces detected.")
    else:
        # read ori img, crop face and save to input_images/cropped_face.png
        read_n_crop_face(ori_img, faces)

        encoder = extract_encoder(siamese_model)
        username, similarity = verify_user(encoder)
        print(username, similarity)

        # Show image with bounding box and label
        show_img(ori_img, faces, username, similarity)
