import cv2
import numpy as np

# Define the HSV color range for orange NEW VID
#lower_bound = np.array([0, 178, 134])   # Lower bound for orange
#upper_bound = np.array([58, 255, 255])  # Upper bound for orange

#PARAMS OLD VID
lower_bound = np.array([146, 111, 224])   # Lower bound for orange
upper_bound = np.array([179, 255, 255])  # Upper bound for orange

# Open the specified video file
video_path = "/home/pablo/Documents/MIR_S2/optimization/old_vid_buoy.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# List to store past positions for tracking trail
trail = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the selected color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours of the masked area
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours :
        # Find the largest contour (assumed to be the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding box around the object
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Compute center position
        center_x = x + w // 2
        center_y = y + h // 2
        center = (center_x, center_y)

        # Compute diagonal length of bounding box
        diagonal_length = int(np.sqrt(w**2 + h**2))

        # Store center position for trail
        trail.append(center)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw tracking trail
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], (255, 0, 0), 2)

        # Print coordinates and diagonal size
        print(f"Center: ({center_x}, {center_y}), Diagonal Length: {diagonal_length}")

    # Show the result
    cv2.imshow('Orange Object Tracking', frame)

    # Press 'q' to exit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
