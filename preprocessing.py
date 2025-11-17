
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
img = X_train[0]

plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

#STEP 1: Center Cropping (20×20 crop)
def center_crop(image, size=20):
    h, w = image.shape
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    return image[start_y:start_y+size, start_x:start_x+size]

cropped = center_crop(img)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Before Crop")

plt.subplot(1,2,2)
plt.imshow(cropped, cmap='gray')
plt.title("Center Cropped (20×20)")
plt.show()

#STEP 2: Padding (Adding Black Border)

padded = cv2.copyMakeBorder(
    img,
    5, 5, 5, 5,
    cv2.BORDER_CONSTANT,
    value=0
)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(padded, cmap='gray')
plt.title("Padded Image (Border Added)")
plt.show()

#STEP 3: Brightness Adjustment

brightness_img = cv2.convertScaleAbs(img, alpha=1, beta=40)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(brightness_img, cmap='gray')
plt.title("Brightness Increased")
plt.show()

#STEP 4: Contrast Adjustment
contrast_img = cv2.convertScaleAbs(img, alpha=1.7, beta=0)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(contrast_img, cmap='gray')
plt.title("Contrast Increased")
plt.show()

#STEP 5: Horizontal Flip

flipped = cv2.flip(img, 1)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(flipped, cmap='gray')
plt.title("Horizontally Flipped")
plt.show()

from google.colab import drive
drive.mount('/content/drive')