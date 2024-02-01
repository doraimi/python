import cv2
import numpy as np

# Read an image
image = cv2.imread("example_image.jpg")

# Convert the image from BGR to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for green color in HSV format
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# Create a binary mask for green color
mask = cv2.inRange(hsv, lower_green, upper_green)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and find the position of green areas
for contour in contours:
    # Get the center of each contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Draw a circle at the center of each green area
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

# Display the original and marked images
cv2.imshow("Original Image", image)
cv2.imshow("Green Areas", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()