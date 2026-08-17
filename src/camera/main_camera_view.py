import cv2

camera_id = 4

cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print(f"Camera {camera_id} open failed")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame read failed")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()