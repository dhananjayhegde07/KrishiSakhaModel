import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Data directories
dataset_dir = 'plantvillage_dataset/Sugarcane_Leaf_Dataset'  # Change this to your dataset path

# Image size and other constants
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 20  # Increased number of epochs for fine-tuning

# Load images using ImageDataGenerator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    channel_shift_range=30.0,  # Reduced channel shift for faster training
    fill_mode='nearest'
)

# Training and validation data generators
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Get the number of classes
num_diseases = train_generator.num_classes

# Load the pre-trained VGG16 model and freeze the base layers
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
for layer in vgg16_base.layers:
    layer.trainable = False

# Add custom classification layers on top of VGG16
x = Flatten()(vgg16_base.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)  # Regularization to prevent overfitting
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_diseases, activation='softmax')(x)

# Create the final model
model = Model(inputs=vgg16_base.input, outputs=output)

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy','Precision', 'Recall', 'AUC'])

# Model summary
model.summary()

# Set up callbacks for better training control
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model_wheat_vgg16_sugarcane.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# Train the model using the generators
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model on validation data
print("Evaluating the model...")
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Save the final model
model.save('crop_disease_model_sugarcane_vgg16_ivattu.keras')
print("Model saved successfully.")
