"""Simple raw camera test - no AI, just shows your live webcam."""
import cv2, sys

print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    sys.exit(1)

print(f"Camera OK! Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
print("You should see your face/surroundings in the window.")
print("Press Q to quit.")

cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Test", 640, 480)

while True:
    ret, frame = cap.read()
    if ret and frame is not None:
        cv2.putText(frame, "CAMERA WORKING! Hold leaf to test.", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera Test", frame)
    else:
        print("No frame!")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
