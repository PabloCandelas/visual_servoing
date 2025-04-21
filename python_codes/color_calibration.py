# PDCP Color calibration for object tracking
# Last update 21/abril/2025

import cv2
import numpy as np

print('\nHola!!\n')
print("   Move the sliders to adjust the HSV values to correctly define the object \n\t-click on the video to see the HSV of the pixel\n\t-press 'p' to print the currect HSV slider values\n\t-press 'q' to exit")

def nothing(x):
    pass

# HSV values on click
def get_hsv_on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_image = param
        if y < hsv_image.shape[0] and x < hsv_image.shape[1]:
            hsv_value = hsv_image[y, x]
            print(f"Clicked HSV at ({x},{y}): H={hsv_value[0]}, S={hsv_value[1]}, V={hsv_value[2]}")



# Create a separate window for sliders
cv2.namedWindow("Color Tuner")

# Create trackbars for color tuning
cv2.createTrackbar("Lower H", "Color Tuner", 0, 179, nothing)
cv2.createTrackbar("Lower S", "Color Tuner", 0, 255, nothing)
cv2.createTrackbar("Lower V", "Color Tuner", 0, 255, nothing)
cv2.createTrackbar("Upper H", "Color Tuner", 179, 179, nothing)
cv2.createTrackbar("Upper S", "Color Tuner", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Color Tuner", 255, 255, nothing)

# Open video file or webcam
video_path = "phone_objects_videos/orange_buoy.mp4"  # Change to your video file or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get FPS for playback speed control
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30  # Default to 30ms if FPS is unknown

# Get the original aspect ratio
_, frame = cap.read()
original_height, original_width = frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video

# Define maximum display width for all images
max_display_width = 600  # Change this if needed

# Compute the height that maintains the aspect ratio
aspect_ratio = original_height / original_width
display_width = max_display_width // 2  # Each image takes half the total width
display_height = int(display_width * aspect_ratio)  # Maintain aspect ratio

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
        continue

    # Resize frame while maintaining aspect ratio
    frame = cv2.resize(frame, (display_width, display_height))

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Click of mouse
    cv2.setMouseCallback("Color Calibration", get_hsv_on_click, hsv)

    # Get trackbar positions
    lower_h = cv2.getTrackbarPos("Lower H", "Color Tuner")
    lower_s = cv2.getTrackbarPos("Lower S", "Color Tuner")
    lower_v = cv2.getTrackbarPos("Lower V", "Color Tuner")
    upper_h = cv2.getTrackbarPos("Upper H", "Color Tuner")
    upper_s = cv2.getTrackbarPos("Upper S", "Color Tuner")
    upper_v = cv2.getTrackbarPos("Upper V", "Color Tuner")
    
    lower_color = np.array([lower_h, lower_s, lower_v])
    upper_color = np.array([upper_h, upper_s, upper_v])
    
    # Create mask and apply it
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert mask to BGR so all images have 3 channels
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Stack images into a 2x2 grid (Original + Mask | Filtered + Empty Space)
    top_row = np.hstack((frame, mask_bgr))  # Original + Mask
    bottom_row = np.hstack((result, np.zeros_like(result)))  # Filtered + Empty space
    combined_display = np.vstack((top_row, bottom_row))  # Stack rows vertically

    # Show the combined display in a single window
    cv2.imshow("Color Calibration", combined_display)
    
    key = cv2.waitKey(delay) & 0xFF

    if key == ord('q'): #quit the program
        break
    elif key == ord('p'): #print the values
        print("\nCurrent HSV Slider Values:")
        print(f"Lower H: {lower_h}, Lower S: {lower_s}, Lower V: {lower_v}")
        print(f"Upper H: {upper_h}, Upper S: {upper_s}, Upper V: {upper_v}")

# Print final trackbar values on exit
print("\nFinal HSV Values:")
print(f"Lower H: {lower_h}, Lower S: {lower_s}, Lower V: {lower_v}")
print(f"Upper H: {upper_h}, Upper S: {upper_s}, Upper V: {upper_v}")

cap.release()
cv2.destroyAllWindows()

print("\n\tBYEEE\n")