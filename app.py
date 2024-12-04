# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### FIRST WE NEED TO IMPORT THE DATA AND TRAIN THE CNN

# +

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\train")
test_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\test")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
            

# +

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\train")
test_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\test")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
            

# +
image_size = (224, 224)

def resize_image(image, size=image_size):
    return cv2.resize(image, size)



# -

def normalize_image(image):
    return image / 255.0



# +
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)


# +
from sklearn.preprocessing import LabelEncoder

labels = ['normal', 'pneumonia', 'covid-19', 'normal', 'covid-19', 'pneumonia']

label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(labels)

print(f"Encoded Labels: {encoded_labels}")


# +

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\train")
test_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\test")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
            

# +

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\train")
test_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\test")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
            
# -

print(train_generator.class_indices)


# +
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# -

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)


# +
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# -

model.save('covid_pneumonia_classifier.h5')


# +

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\train")
test_dir = os.path.join("C:\\Users\\Kiran gowda.A\\Downloads\\archive\\Data\\test")

if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory not found: {test_dir}")

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
            
# -

test_generator = datagen.flow_from_directory(
    test_dir,  
    target_size=(150, 150), 
    batch_size=32,
    class_mode='categorical',
    shuffle=False  
)


import numpy as np

# +

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)  

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())  

from sklearn.metrics import classification_report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# -

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# +
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('covid_pneumonia_classifier.h5')  # Load your trained model
class_indices = {0: "COVID", 1: "Pneumonia", 2: "Normal"}  # Adjust as per your classes

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    # Load and preprocess the image
    file = request.files['file']
    img = image.load_img(file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_indices[np.argmax(predictions)]
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)

# -


