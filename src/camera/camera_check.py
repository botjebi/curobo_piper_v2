import cv2

working_cameras = []

for i in range(6):
    print(f"Checking camera {i}...")

    cap = cv2.VideoCapture(i)

    if not cap.isOpened():
        print(f"  camera {i}: open failed")
        cap.release()
        continue

    ret, frame = cap.read()

    if not ret:
        print(f"  camera {i}: opened, but frame read failed")
        cap.release()
        continue

    print(f"  camera {i}: SUCCESS")
    print(f"  resolution: {frame.shape[1]} x {frame.shape[0]}")

    working_cameras.append(i)
    cap.release()

print()
print("Available cameras:", working_cameras)