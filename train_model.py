from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, 
Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import os 
 
# Set parameters 
img_height, img_width = 64, 64 
batch_size = 32 
num_classes = 43  # For GTSRB 
 
# Dataset directories 
train_dir = 'data/train' 
val_dir = 'data/val' 
 
# Preprocess the data 
train_datagen = ImageDataGenerator(rescale=1./255) 
val_datagen = ImageDataGenerator(rescale=1./255) 
 
train_generator = train_datagen.flow_from_directory( 
    train_dir, 
    target_size=(img_height, img_width), 
    batch_size=batch_size, 
    class_mode='categorical' 
) 
9 
 
val_generator = val_datagen.flow_from_directory( 
    val_dir, 
    target_size=(img_height, img_width), 
    batch_size=batch_size, 
    class_mode='categorical' 
) 
 
# Build CNN Model 
model = Sequential([ 
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)), 
    MaxPooling2D(2,2), 
    Conv2D(64, (3,3), activation='relu'), 
    MaxPooling2D(2,2), 
    Flatten(), 
    Dense(256, activation='relu'), 
    Dropout(0.5), 
    Dense(num_classes, activation='softmax') 
]) 
 
model.compile(optimizer='adam', loss='categorical_crossentropy', 
metrics=['accuracy']) 
 
# Train 
model.fit(train_generator, epochs=10, validation_data=val_generator) 
 
# Save model 
model.save('traffic_sign_model.h5') 
