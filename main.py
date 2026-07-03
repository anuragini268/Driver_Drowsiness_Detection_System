import cv2
import mediapipe as mp
import winsound
from src.ear import eye_aspect_ratio

# -------------------------------
# Initialize Face Mesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------
# Open Webcam
# -------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam Opened Successfully")

# -------------------------------
# Eye Landmark IDs
# -------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------------------------------
# Thresholds
# -------------------------------
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20

COUNTER = 0
ALARM_ON = False

# -------------------------------
# Main Loop
# -------------------------------
while True:

    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            left_eye = []
            right_eye = []

            # Left Eye
            for idx in LEFT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                left_eye.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Right Eye
            for idx in RIGHT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                right_eye.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # -------------------------------
            # Calculate EAR
            # -------------------------------
            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)

            ear = (leftEAR + rightEAR) / 2.0

            cv2.putText(
                frame,
                f"EAR : {ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # -------------------------------
            # Drowsiness Detection
            # -------------------------------
            if ear < EAR_THRESHOLD:

                COUNTER += 1

                cv2.putText(
                    frame,
                    "Eyes Closed",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                if COUNTER >= FRAME_THRESHOLD:

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    if not ALARM_ON:
                        winsound.Beep(1000, 1000)
                        ALARM_ON = True

            else:

                COUNTER = 0
                ALARM_ON = False

                cv2.putText(
                    frame,
                    "Eyes Open",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                frame,
                f"Counter : {COUNTER}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    else:

        cv2.putText(
            frame,
            "Face Not Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Driver Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()