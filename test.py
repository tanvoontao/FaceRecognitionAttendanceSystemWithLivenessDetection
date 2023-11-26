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
INPUT_IMG_DIR = 'input_images/face.png'
THRESHOLD = 0.5
VERIFICATION_THRESHOLD = 0.5

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
        
        predictions = [1 if sim > THRESHOLD else 0 for sim in similarities]

        similar_pairs_count = sum(predictions)

        similarity = similar_pairs_count / len(img_pairs) if img_pairs else 0

        user = {
            'username': current_user,
            'similarity': similarity
        }

        users.append(user)

    possible_user = max(users, key=lambda user: user['similarity'])

    username = possible_user['username']
    similarity = possible_user['similarity']

    if (similarity > VERIFICATION_THRESHOLD):
        return username, similarity
    else:
        return None, None

encoder = extract_encoder(siamese_model)

username, similarity = verify_user(encoder)
print(username, similarity)