# Image Classifier using CNN: Part 2

### Step 6: Train the Model
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

from numpy import argmax

# Load saved model
model = load_model('classifier.keras')

# Load saved dataset
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

class_names = np.load('class_names.npy', allow_pickle=True)
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test))
print('The model has successfully trained')

model.save('classifier.keras')
print('Saving the model as classifier.keras')

### Step 7: Evaluate the Model

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

### Step 8: Classify a New Image

from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]

### Step 9: Predict What's in the Image

def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def run_example():
    img = load_image(filename)
    loaded_model = load_model('classifier.keras')
    result = loaded_model.predict(img)
    predicted_index = argmax(result[0])
    print('Predicted class:', class_names[predicted_index].title())

run_example()