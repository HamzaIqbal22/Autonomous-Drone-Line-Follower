import cv2
import os
import time
from threading import Thread
from djitellopy import Tello

tello = Tello()

tello.connect()
tello.get_battery()
keepStreaming = True
tello.streamon()

photo_count = 800  # Variable to keep track of the number of photos taken
output_dir = "photos"  # Directory to store the photos

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def videoStreamer():
    global keepStreaming, frame
    while keepStreaming:
        try:
            frame = tello.get_frame_read().frame
            cv2.imshow('Tello Stream', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):  # Press 'q' to quit
                break
        except Exception as e:
            print("Error during streaming:", e)
            continue

    cv2.destroyAllWindows()

# Function to take photos
def take_photos(num_photos):
    global photo_count, frame
    for i in range(num_photos):
        photo_count += 1
        filename = os.path.join(output_dir, f"photo_{photo_count}.jpg")
        cv2.imwrite(filename, frame)  # Save the frame as an image
        print(f"Photo {photo_count} saved as {filename}")
        cv2.waitKey(100)  # Wait for 100ms before taking the next photo

# Run the streamer in a separate thread
streamer = Thread(target=videoStreamer)
streamer.start()

try:
    while True:
        input_val = input("Press Enter to take the next 100 photos (or 'q' to quit): ")
        if input_val.strip().lower() == 'q':
            break
        if keepStreaming:
            take_photos(100)
        else:
            break
except KeyboardInterrupt:
    pass  # Handle Ctrl+C gracefully

# End the streaming thread
keepStreaming = False
streamer.join()
tello.streamoff()  # Turn off the stream
