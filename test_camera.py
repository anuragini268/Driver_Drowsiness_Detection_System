import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow("Camera", frame)
    else:
        print("Cannot read frame")

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()