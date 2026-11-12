
# CNN Image Classifier
# Complete Activity: Part 1 + Part 2
# For VS Code (Windows)

# -----------------------------------------
# Step 1: Import Libraries
# -----------------------------------------

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical, load_img, img_to_array
from tensorflow.keras.optimizers import SGD

from numpy import argmax
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import tarfile


# -----------------------------------------
# Step 2: Download the Dataset
# -----------------------------------------

url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"

if not os.path.exists("cifar10"):
    print("Downloading dataset...")

    urllib.request.urlretrieve(url, "cifar10.tgz")

    print("Download completed!")

    print("Extracting dataset...")

    with tarfile.open("cifar10.tgz", "r:gz") as tar:
        tar.extractall(".")

    print("Dataset extracted successfully!")

else:
    print("Dataset already exists!")


# -----------------------------------------
# Step 3: Load Images into Arrays
# -----------------------------------------

def load_split(folder):

    class_names = sorted(os.listdir(folder))

    images = []
    labels = []

    for label, class_name in enumerate(class_names):

        class_dir = os.path.join(folder, class_name)

        for filename in os.listdir(class_dir):

            img = load_img(
                os.path.join(class_dir, filename),
                color_mode='rgb'
            )

            images.append(img_to_array(img))
            labels.append(label)

    x = np.array(images)
    y = np.array(labels).reshape(-1, 1)

    # Shuffle images randomly
    shuffle_order = np.random.permutation(len(x))

    return x[shuffle_order], y[shuffle_order], class_names


x_train, y_train, class_names = load_split('cifar10/train')

x_test, y_test, _ = load_split('cifar10/test')

print("Classes:", class_names)


# Display 9 sample images

for i in range(9):

    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i].astype('uint8'))
    plt.axis('off')

plt.show()


# -----------------------------------------
# Step 4: Preprocess the Dataset
# -----------------------------------------

num_classes = len(class_names)

train_size = 8000
test_size = 2000

x_train = x_train[:train_size]
y_train = y_train[:train_size]

x_test = x_test[:test_size]
y_test = y_test[:test_size]

# Convert labels into categorical format

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Normalize pixel values between 0 and 1

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

print("Training samples:", x_train.shape[0])
print("Testing samples:", x_test.shape[0])


# -----------------------------------------
# Step 5: Build the CNN Model
# -----------------------------------------

model = Sequential()

model.add(Input(shape=(32, 32, 3)))

model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(128, activation='relu',
                kernel_constraint=max_norm(3)))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

# Compile the model

opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=False)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

model.summary()


# -----------------------------------------
# Step 6: Train the Model
# -----------------------------------------

model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test)
)

print("The model has successfully trained!")

# Save the trained model

model.save('classifier.keras')

print("Model saved as classifier.keras")


# -----------------------------------------
# Step 7: Evaluate the Model
# -----------------------------------------

score = model.evaluate(x_test, y_test, verbose=0)

print("Test Loss:", score[0])
print("Test Accuracy:", score[1])


# -----------------------------------------
# Step 8: Select a New Image
# -----------------------------------------

# Enter the image path from your computer

filename = input("Enter the image path: ")


# -----------------------------------------
# Step 9: Predict What's in the Image
# -----------------------------------------

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

    print("Predicted class:",
          class_names[predicted_index].title())


run_example()
