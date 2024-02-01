import cv2
import datetime

# Open the camera (camera index 0 typically refers to the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture a single frame
ret, frame = cap.read()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not capture frame.")
    cap.release()
    exit()

# Get the current date and time
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")

# Create a filename with the timestamp
filename = f"captured_photo_{timestamp}.jpg"

# Save the captured frame with timestamp in the filename
cv2.imwrite(filename, frame)

# Release the camera
cap.release()

print(f"Photo captured and saved as {filename}.")
