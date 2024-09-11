import cv2
import numpy as np

# Constants
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_COLOR = (0, 255, 0)  # Green tape color assumed

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Define the coordinates of the 16:9 framed area on the wall (adjust based on setup)
physical_frame = np.array([[100, 100], [1180, 100], [1180, 620], [100, 620]], dtype=np.float32)

# Define the virtual screen coordinates (mapped to a full-size 16:9 screen)
virtual_frame = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT], [0, FRAME_HEIGHT]], dtype=np.float32)

# Calculate the homography matrix to map points from the physical frame to the virtual screen
homography_matrix, _ = cv2.findHomography(physical_frame, virtual_frame)

# Create a virtual screen (white background)
virtual_screen = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255

# Background subtraction for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

def draw_virtual_hit(x, y):
    """Draws the hit point on the virtual screen."""
    global virtual_screen
    cv2.circle(virtual_screen, (int(x), int(y)), 15, (0, 0, 0), -1)

def detect_hit(frame):
    """Detects the ball hit within the framed area and marks the impact point."""
    fgmask = fgbg.apply(frame)

    # Remove noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust based on ball size
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 5:  # Minimum radius to qualify as a hit
                hit_point = np.array([[x, y]], dtype=np.float32)
                hit_virtual = cv2.perspectiveTransform(hit_point[None, :, :], homography_matrix)
                draw_virtual_hit(hit_virtual[0, 0, 0], hit_virtual[0, 0, 1])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    detect_hit(frame)
    
    # Draw the physical frame on the captured frame
    cv2.polylines(frame, [physical_frame.astype(int)], isClosed=True, color=FRAME_COLOR, thickness=3)

    # Display both the webcam frame and the virtual screen
    cv2.imshow('Webcam View', frame)
    cv2.imshow('Virtual Screen', virtual_screen)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
