import cv2
import datetime

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

# Get the current date and time for the filename
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")

# Create the filename with the timestamp
filename = f"captured_video_{timestamp}.avi"

out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

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

print(f"Video captured and saved as {filename}.")
