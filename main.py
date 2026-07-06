import cv2
import mediapipe as mp
import winsound

from src.ear import eye_aspect_ratio
from src.head_pose import get_head_pose

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
    print("Cannot Open Webcam")
    exit()

print("Webcam Started Successfully")

# -------------------------------
# Eye Landmarks
# -------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# -------------------------------
# Mouth Landmarks
# -------------------------------
UPPER_LIP = 13
LOWER_LIP = 14

# -------------------------------
# Thresholds
# -------------------------------
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20
YAWN_THRESHOLD = 25

# -------------------------------
# Counters
# -------------------------------
COUNTER = 0
YAWN_COUNTER = 0
DISTRACTION_COUNTER = 0
FATIGUE_SCORE = 0

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

            # -------------------------------
            # Left Eye
            # -------------------------------
            for idx in LEFT_EYE:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                left_eye.append((x, y))

                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # -------------------------------
            # Right Eye
            # -------------------------------
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
            # Eye Status
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
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
             
            # -------------------------------
            # Yawning Detection
            # -------------------------------

            upper_lip = face_landmarks.landmark[UPPER_LIP]
            lower_lip = face_landmarks.landmark[LOWER_LIP]

            upper_x = int(upper_lip.x * w)
            upper_y = int(upper_lip.y * h)

            lower_x = int(lower_lip.x * w)
            lower_y = int(lower_lip.y * h)

            # Draw Mouth Points
            cv2.circle(frame, (upper_x, upper_y), 3, (255, 0, 255), -1)
            cv2.circle(frame, (lower_x, lower_y), 3, (255, 0, 255), -1)

            # Calculate Mouth Opening
            mouth_open = abs(lower_y - upper_y)

            cv2.putText(
                frame,
                f"Mouth : {mouth_open}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if mouth_open > YAWN_THRESHOLD:

                YAWN_COUNTER += 1
                FATIGUE_SCORE += 1

                cv2.putText(
                    frame,
                    "YAWNING DETECTED!",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                if not ALARM_ON:
                    winsound.Beep(1200, 600)
                    ALARM_ON = True

            else:

                if ear >= EAR_THRESHOLD:
                    ALARM_ON = False

            cv2.putText(
                frame,
                f"Yawns : {YAWN_COUNTER}",
                (420, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            # -------------------------------
            # Head Pose Detection
            # -------------------------------
            head_status = get_head_pose(face_landmarks, w, h)

            cv2.putText(
                frame,
                f"Head : {head_status}",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if head_status != "Forward":

                DISTRACTION_COUNTER += 1
                FATIGUE_SCORE += 1

                cv2.putText(
                    frame,
                    "DRIVER DISTRACTED!",
                    (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                if not ALARM_ON:
                    winsound.Beep(1500, 700)
                    ALARM_ON = True

            # -------------------------------
            # Drowsiness Alert
            # -------------------------------
            if COUNTER >= FRAME_THRESHOLD:

                FATIGUE_SCORE += 2

                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                if not ALARM_ON:
                    winsound.Beep(1800, 1000)
                    ALARM_ON = True

            # -------------------------------
            # Fatigue Score
            # -------------------------------
            cv2.putText(
                frame,
                f"Fatigue Score : {FATIGUE_SCORE}",
                (20, 360),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # -------------------------------
            # Driver Analytics
            # -------------------------------
            cv2.putText(
                frame,
                f"Distractions : {DISTRACTION_COUNTER}",
                (420, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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