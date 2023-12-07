import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2

current_directory = os.getcwd()

dataset_dir = f'{current_directory}/final'
train_dir = f'{current_directory}/final/train'
test_dir = f'{current_directory}/final/test'

import json
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
    Input,

)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE  = (160, 160)

train_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  brightness_range=(0.8,1.2),
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  fill_mode='nearest',
  shear_range=0.2,
  zoom_range=0.3,
  # rescale=1./255
)
valid_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,
  # rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
  train_dir,target_size=IMG_SIZE,
  color_mode='rgb',
  class_mode='binary',
  batch_size=25,
  shuffle=True
)


valid_generator = valid_datagen.flow_from_directory(
  test_dir,
  target_size=IMG_SIZE,
  color_mode='rgb',
  class_mode='binary',
  batch_size=25
)

# -------------------- base model + feature extraction -------------------- #
IMG_SHAPE = IMG_SIZE + (3,) # (160, 160, 3)

base_model = MobileNetV2(
  weights="imagenet",
  include_top=False,
  input_tensor=Input(shape=IMG_SHAPE)
)

base_model.trainable = False

base_model.summary()

# -------------------- model -------------------- #

output = Flatten()(base_model.output)
output = Dropout(0.3)(output)
output = Dense(units = 8,activation='relu')(output)
prediction = Dense(1,activation='sigmoid')(output)

model = Model(inputs = base_model.input,outputs = prediction)
model.summary()

# -------------------- callbacks -------------------- #
early_stopping = EarlyStopping(
  monitor='val_loss',
  patience=5,
  verbose=1,
  restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
  'best_model.h5',
  monitor='val_loss',
  verbose=1,
  save_best_only=True
)


# tell the model what cost and optimization method to use
model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(
    learning_rate=0.000001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
  ),
  metrics=['accuracy']
)

# (extra)
loss0, accuracy0 = model.evaluate(valid_generator)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
# (extra)


history = model.fit(
  train_generator,
  steps_per_epoch = train_generator.samples // 25,
  validation_data = valid_generator, 
  validation_steps = valid_generator.samples // 25,
  epochs = 100,
  callbacks=[early_stopping, model_checkpoint]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(history.history['loss'])+1])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

history_dict = history.history
history_json_file = f'{current_directory}/initial_training_history.json'

with open(history_json_file, 'w') as f:
    json.dump(history_dict, f)



# fine tuning

with open('initial_training_history.json', 'r') as f:
    initial_history = json.load(f)

INIT_EPOCHS = 100
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS =  INIT_EPOCHS + FINE_TUNE_EPOCHS


base_model = load_model('best_model.h5')

print("Number of layers in the base model: ", len(base_model.layers))

base_model.trainable = True

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

base_model.compile(
    optimizer=Adam(
        learning_rate=0.0000005,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    ),
    loss='binary_crossentropy',
    metrics = ['accuracy']
)


early_stopping = EarlyStopping(
  monitor='val_loss',
  patience=5,
  verbose=1,
  restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
  'best_model_fine_tune.h5',
  monitor='val_loss',
  verbose=1,
  save_best_only=True
)

history_fine = base_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 25,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // 25,
    epochs=TOTAL_EPOCHS,  # End epoch
    initial_epoch=INIT_EPOCHS,  # Start epoch
    callbacks=[early_stopping, model_checkpoint]
)

acc = initial_history['accuracy'] + history_fine.history['accuracy']
val_acc = initial_history['val_accuracy'] + history_fine.history['val_accuracy']
loss = initial_history['loss'] + history_fine.history['loss']
val_loss = initial_history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.plot([INIT_EPOCHS-1,INIT_EPOCHS-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, max(initial_history['loss'])+1])
plt.plot([INIT_EPOCHS-1,INIT_EPOCHS-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()



# function use to copy image from ori to final dataset
def train_test_splits(data_directory):
  
  for split_type in os.listdir(data_directory):
    
    path_to_split_type = os.path.join(data_directory,split_type)
    
    for category in os.listdir(path_to_split_type):
      path_to_category = os.path.join(path_to_split_type,category)
      
      for subject in os.listdir(path_to_category):
        path_to_subject = os.path.join(path_to_category,subject)
        
        for img in os.listdir(path_to_subject):
          
          source_path = os.path.join(path_to_subject, img)
          
          if split_type == 'train':
            destination_path = os.path.join(train_dir,category,img)
          else:
            destination_path = os.path.join(test_dir,category,img)
          
          shutil.copy(source_path, destination_path)
          