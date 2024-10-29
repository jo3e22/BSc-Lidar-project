#%%
import cv2
import leddartech
import time

# Initialize the LIDAR
lidar = leddartech.LeddarTech()
lidar.connect()
lidar.start()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get LIDAR data
    data = lidar.get_data()

    # Detect objects in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

    # Display LIDAR data and webcam frame with detected objects
    cv2.imshow("LIDAR Data", data)
    cv2.imshow("Webcam with Objects", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
lidar.stop()
lidar.disconnect()
cv2.destroyAllWindows()

# %%
