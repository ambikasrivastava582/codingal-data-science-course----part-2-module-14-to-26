
import cv2
import numpy as np
import math
from vcam import vcam, meshGen
import matplotlib.pyplot as plt

# ----------------------------------
# PART 1: Funny Mirror Effect
# ----------------------------------

plt.figure(figsize=(20, 20))

# Read the input image
# Keep minions.jpg in the same folder as this Python file
img = cv2.imread("minions.jpg")

if img is None:
    print("Image not found! Please check minions.jpg")
    exit()

H, W = img.shape[:2]

# Creating the virtual camera object
c1 = vcam(H=H, W=W)

# Creating the surface object
plane = meshGen(H, W)

# Generate a mirror effect using Z coordinate
plane.Z += 20 * np.exp(
    -0.5 * ((plane.X * 1.0 / plane.W) / 0.1) ** 2
) / (0.1 * np.sqrt(2 * np.pi))

pts3d = plane.getPlane()

pts2d = c1.project(pts3d)
map_x, map_y = c1.getMaps(pts2d)

output = cv2.remap(
    img, map_x, map_y,
    interpolation=cv2.INTER_LINEAR
)

plt.subplot(1, 2, 1)
plt.title("Funny Mirror - X Direction")

plt.imshow(
    cv2.cvtColor(
        np.hstack((img, output)),
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.show()


# ----------------------------------
# PART 2: Another Funny Mirror Effect
# ----------------------------------

plt.figure(figsize=(20, 20))

# Reading the second input image
img = cv2.imread("minions.jpg")

if img is None:
    print("Image not found! Please check minions.jpg")
    exit()

H, W = img.shape[:2]

# Creating the virtual camera object
c1 = vcam(H=H, W=W)

# Creating the surface object
plane = meshGen(H, W)

# Generate another mirror effect using Y coordinate
plane.Z += 20 * np.exp(
    -0.5 * ((plane.Y * 1.0 / plane.H) / 0.1) ** 2
) / (0.1 * np.sqrt(2 * np.pi))

pts3d = plane.getPlane()

pts2d = c1.project(pts3d)
map_x, map_y = c1.getMaps(pts2d)

output = cv2.remap(
    img, map_x, map_y,
    interpolation=cv2.INTER_LINEAR
)

plt.subplot(1, 2, 1)
plt.title("Funny Mirror - Y Direction")

plt.imshow(
    cv2.cvtColor(
        np.hstack((img, output)),
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.show()
