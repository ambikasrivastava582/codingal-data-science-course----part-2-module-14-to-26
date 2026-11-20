
import numpy as np  # Linear algebra
import pandas as pd  # Data processing
import cv2
import matplotlib.pyplot as plt


# =================================
# Creating a Blank Image
# =================================

# Create a black image of 512 x 512 pixels
emptyImg = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

# Display blank image
plt.imshow(emptyImg)
plt.title("Blank Image")
plt.show()


# =================================
# Creating a Rectangle
# =================================

cv2.rectangle(
    emptyImg,
    pt1=(380, 10),
    pt2=(500, 150),
    color=(0, 255, 0),
    thickness=10
)

plt.imshow(emptyImg)
plt.title("Rectangle")
plt.show()


# =================================
# Creating a Filled Square
# =================================

cv2.rectangle(
    emptyImg,
    pt1=(200, 200),
    pt2=(300, 300),
    color=(0, 0, 255),
    thickness=-1
)

plt.imshow(emptyImg)
plt.title("Filled Square")
plt.show()


# =================================
# Creating a Circle
# =================================

cv2.circle(
    emptyImg,
    center=(100, 100),
    radius=50,
    color=(255, 0, 0),
    thickness=10
)

plt.imshow(emptyImg)
plt.title("Circle")
plt.show()


# =================================
# Creating a Filled Circle
# =================================

cv2.circle(
    emptyImg,
    center=(400, 400),
    radius=50,
    color=(255, 0, 0),
    thickness=-1
)

plt.imshow(emptyImg)
plt.title("Filled Circle")
plt.show()


# =================================
# Creating a Linear Line
# =================================

cv2.line(
    emptyImg,
    pt1=(0, 0),
    pt2=(512, 512),
    color=(255, 255, 255),
    thickness=10
)

plt.imshow(emptyImg)
plt.title("Line")
plt.show()


# =================================
# Creating Another Blank Image
# =================================

emptyImg2 = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

plt.imshow(emptyImg2)
plt.title("Blank Image 2")
plt.show()


# =================================
# Adding Text on the Image
# =================================

cv2.putText(
    emptyImg2,
    text='Image',
    org=(20, 250),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=4,
    color=(255, 255, 255),
    thickness=3,
    lineType=cv2.LINE_AA
)

plt.imshow(emptyImg2)
plt.title("Text on Image")
plt.show()
