import math

# MediaPipe FaceMesh Landmark IDs
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
CHIN = 152
FOREHEAD = 10


def get_head_pose(face_landmarks, width, height):

    nose = face_landmarks.landmark[NOSE_TIP]
    left = face_landmarks.landmark[LEFT_CHEEK]
    right = face_landmarks.landmark[RIGHT_CHEEK]
    chin = face_landmarks.landmark[CHIN]
    forehead = face_landmarks.landmark[FOREHEAD]

    nose_x = nose.x * width
    nose_y = nose.y * height

    left_x = left.x * width
    right_x = right.x * width

    chin_y = chin.y * height
    forehead_y = forehead.y * height

    face_center_x = (left_x + right_x) / 2

    # -------------------------
    # LEFT / RIGHT
    # -------------------------
    horizontal_offset = nose_x - face_center_x

    if horizontal_offset < -20:
        return "Left"

    elif horizontal_offset > 20:
        return "Right"

    # -------------------------
    # UP / DOWN
    # -------------------------
    face_height = chin_y - forehead_y

    if face_height < 140:
        return "Up"

    elif face_height > 220:
        return "Down"

    return "Forward"