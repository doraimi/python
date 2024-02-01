import cv2
import datetime

"""    
To capture a video for a specific duration (e.g., 5 seconds) from the camera and save it, 
you can modify the code to capture frames continuously for the specified duration and then save them as a video. 
Here's how you can do it:
"""

# Open the camera (camera index 0 typically refers to the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('captured_video.avi', fourcc, 20.0, (frame_width, frame_height))

# Get the current time to calculate the end time
start_time = datetime.datetime.now()
end_time = start_time + datetime.timedelta(seconds=5)  # Capture video for 5 seconds

# Capture video for the specified duration
while datetime.datetime.now() < end_time:
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Could not capture frame.")
        break

    # Write the frame to the output video
    out.write(frame)

    # Display the captured frame (optional)
    cv2.imshow('Capturing Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and video writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Video captured successfully.")

"""
In this modified code:

The VideoWriter object is created to save the captured frames as a video (captured_video.avi) using the XVID codec.
Frames are captured continuously until the specified duration (5 seconds) is reached.
The captured frames are written to the output video using the write method of the VideoWriter object.
The program waits for a key press to end capturing the video. Press 'q' to stop capturing.
After the specified duration, the camera and video writer are released, and OpenCV windows are closed.
Run this script, and it will capture a video from your camera for 5 seconds and save it as "captured_video.avi". Adjust the codec, frame rate, and other parameters as needed.
"""