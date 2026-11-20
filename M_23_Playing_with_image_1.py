
# ==========================================
# OPEN CV BASICS
# ==========================================

# Importing Libraries

# OpenCV library
import cv2

# For matrix manipulation
import numpy as np


# ==========================================
# CHANGING IMAGE'S COLOR PROFILES
# ==========================================

# Read the image
# Keep image.jpg in the same folder as Python file
image = cv2.imread("lion.jpg")

# Check whether image is loaded successfully
if image is None:
    print("Error: Image not found. Check the image path.")
    exit()

# Display original image
cv2.imshow("Original Image", image)

# Print dimensions
print("Original Image Dimensions:", image.shape)


# Convert color image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display grayscale image
cv2.imshow("Grayscale Image", gray_image)

print("Grayscale Image Dimensions:", gray_image.shape)


# Convert color image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Display HSV image
cv2.imshow("HSV Image", hsv_image)

print("HSV Image Dimensions:", hsv_image.shape)


# ==========================================
# EDGE DETECTION
# ==========================================

# Canny image is an edge-detected image
# It takes grayscale image as an argument

canny_image = cv2.Canny(gray_image, 150, 200)

cv2.imshow("Canny Edge Detection", canny_image)


# Erosion (used to reduce small white regions)
kernel = np.ones((1, 1), np.uint8)

# Small window or matrix
erode_image = cv2.erode(
    canny_image, kernel, iterations=1
)

cv2.imshow("Erosion Image", erode_image)


# Dilation
kernel1 = np.ones((5, 5), np.uint8)

dilate_image = cv2.dilate(
    canny_image, kernel1, iterations=1
)

cv2.imshow("Dilation Image", dilate_image)


# Displaying all three images in one row
hori_display = np.concatenate(
    (canny_image, erode_image, dilate_image), axis=1
)

cv2.imshow("Edge Detection - Erosion - Dilation", hori_display)


# ==========================================
# IMAGE MANIPULATION - DENOISING
# ==========================================

# Read noisy image
# You can use another image named noisy_image.jpg
image = cv2.imread("color.jpg")

# Check whether image is loaded successfully
if image is None:
    print("Error: Noisy image not found. Check the image path.")

else:

    # Make sure image is in the correct format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # Denoise image
    denoise_image = cv2.fastNlMeansDenoisingColored(
        image, None, 20, 20, 7, 15
    )


    # Display original and denoised images
    hori_display = np.concatenate(
        (image, denoise_image), axis=1
    )

    cv2.imshow("Original and Denoised Images", hori_display)


# Wait until a key is pressed
cv2.waitKey(0)

# Close all image windows
cv2.destroyAllWindows()
