import cv2

# Load the Haar Cascade file
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Read the image
img = cv2.imread("face.jpg")

if img is None:
    print("Image not found. Check the image path.")
    exit()

# Convert image into grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Display output
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
