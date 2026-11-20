
import cv2
import numpy as np

# =====================================
# UPLOADING / READING IMAGE
# =====================================

# Read image
# Keep your image in the same folder as Python file
img = cv2.imread("shapes.png")

# Check whether image is loaded
if img is None:
    print("Error: Image not found. Check the image path.")
    exit()

# Display original image
cv2.imshow("Original Image", img)


# =====================================
# CONVERTING BGR IMAGE TO HSV IMAGE
# =====================================

# Converting BGR image to HSV image
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("HSV Image", hsv_img)


# =====================================
# DETECTING BLUE COLOR
# =====================================

# Specifying range of blue color
lower_blue = np.array([65, 0, 0])
upper_blue = np.array([110, 255, 255])

# Creating mask for only blue color
mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

cv2.imshow("Blue Color Mask", mask)

# Returning result (mask) in blue color except the black one
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Blue Color Detection", result)


# =====================================
# DETECTING RED COLOR
# =====================================

# Specifying range of red color
lower_red = np.array([0, 100, 20])
upper_red = np.array([10, 255, 255])

# Creating mask for only red color
mask = cv2.inRange(hsv_img, lower_red, upper_red)

cv2.imshow("Red Color Mask", mask)

# Returning result (mask) in red color except the black one
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Red Color Detection", result)


# =====================================
# DETECTING YELLOW COLOR
# =====================================

# Specifying range of yellow color
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Creating mask for only yellow color
mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

cv2.imshow("Yellow Color Mask", mask)

# Returning result (mask) in yellow color except the black one
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Yellow Color Detection", result)


# =====================================
# DETECTING GREEN COLOR
# =====================================

# Specifying range of green color
lower_green = np.array([50, 100, 100])
upper_green = np.array([75, 255, 255])

# Creating mask for only green color
mask = cv2.inRange(hsv_img, lower_green, upper_green)

cv2.imshow("Green Color Mask", mask)

# Returning result (mask) in green color except the black one
result = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Green Color Detection", result)


# =====================================
# WAIT AND CLOSE WINDOWS
# =====================================

# Wait until a key is pressed
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
