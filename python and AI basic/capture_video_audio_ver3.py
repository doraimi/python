import cv2
import numpy as np
import pyaudio
import wave
import datetime

# Video settings
frame_width = 640
frame_height = 480
fps = 30

# Audio settings
audio_format = pyaudio.paInt16
channels = 1
sample_rate = 44100
chunk_size = 1024

# Open the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Open the microphone
audio = pyaudio.PyAudio()
stream = audio.open(format=audio_format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_out = cv2.VideoWriter(f"captured_video_{current_time}.avi", fourcc, fps, (frame_width, frame_height))

# Get the current time to calculate the end time
start_time = datetime.datetime.now()
end_time = start_time + datetime.timedelta(seconds=5)  # Capture video for 5 seconds

# Capture audio and video
#while datetime.datetime.now() < end_time:
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Read audio data
    audio_data = stream.read(chunk_size)

    # Write video frame
    video_out.write(frame)

    # Display the captured frame (optional)
    cv2.imshow('Capturing Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera, microphone, and video writer
cap.release()
stream.stop_stream()
stream.close()
audio.terminate()
video_out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
