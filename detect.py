import cv2
from ultralytics import YOLO

# 1. Load your custom model
# Ensure 'best.pt' is in the same directory as this script
model = YOLO("best.pt")

# 2. Initialize the webcam
# '0' is usually the default built-in camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the application.")

while True:
    # Capture frame-by-frame
    success, frame = cap.read()

    if not success:
        print("Error: Failed to read frame.")
        break

    # 3. Run inference on the frame
    # stream=True is more memory-efficient for video
    results = model(frame, stream=True)

    # 4. Visualize the results on the frame
    for r in results:
        annotated_frame = r.plot()

    # Display the resulting frame
    cv2.imshow("Smart Waste Detection System", annotated_frame)

    # 5. Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()