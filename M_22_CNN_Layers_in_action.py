# CNN Layers in Action: Convolution, Pooling, and Keras

### Step 1: Import Libraries

import numpy as np
import matplotlib.pyplot as plt

### Step 2: Create a Small Image and a Filter

image = np.array([
    [1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0],
])

kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
])

print(image)

### Step 3: Slide the Filter to Get a Feature Map

def convolve(image, kernel):
    k = kernel.shape[0]
    out_size = image.shape[0] - k + 1
    result = np.zeros((out_size, out_size))
    for i in range(out_size):
        for j in range(out_size):
            patch = image[i:i+k, j:j+k]
            result[i, j] = np.sum(patch * kernel)
    return result

feature_map = convolve(image, kernel)
print(feature_map)

### Step 4: Shrink It with Max Pooling

def max_pool(feature_map, size=2):
    out_size = feature_map.shape[0] // size
    result = np.zeros((out_size, out_size))
    for i in range(out_size):
        for j in range(out_size):
            window = feature_map[i*size:(i+1)*size, j*size:(j+1)*size]
            result[i, j] = np.max(window)
    return result

pooled = max_pool(feature_map, size=2)
print(pooled)

### Step 5: Flatten the Pooled Map

flattened = pooled.flatten()
print(flattened)

### Step 6: Build the Same Layers with Keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Input(shape=(6, 6, 1)))
model.add(Conv2D(4, kernel_size=(3, 3), activation='relu'))
print('after Conv2D:', model.output_shape)

### Step 7: Add Pooling, Flattening, and a Dense Layer

model.add(MaxPooling2D(pool_size=(2, 2)))
print('after MaxPooling2D:', model.output_shape)

model.add(Flatten())
print('after Flatten:', model.output_shape)

model.add(Dense(10, activation='softmax'))
print('after Dense:', model.output_shape)

### Step 8: Visualize Every Stage

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image (6x6)')

axes[1].imshow(feature_map, cmap='gray')
axes[1].set_title('After Convolution (4x4)')

axes[2].imshow(pooled, cmap='gray')
axes[2].set_title('After Max Pooling (2x2)')

plt.show()