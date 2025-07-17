import cv2
import mediapipe as mp
import numpy as np

# === Load glasses and mustache with alpha channel ===
glasses_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
mustache_img = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)

# === Overlay PNG with alpha ===
def overlay_image(bg, overlay, x, y, size=None):
    if size:
        overlay = cv2.resize(overlay, size)
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = bg[y:y+h, x:x+w, c] * (1 - alpha) + overlay[:, :, c] * alpha
    return bg

# === Setup MediaPipe FaceMesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, min_detection_confidence=0.5)

# === Start webcam ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            def get_point(i):
                lm = face_landmarks.landmark[i]
                return int(lm.x * w), int(lm.y * h)

            left_eye = get_point(33)
            right_eye = get_point(263)
            nose_tip = get_point(1)
            upper_lip = get_point(13)

            # === Glasses positioning ===
            face_width = abs(right_eye[0] - left_eye[0])
            center_x = (right_eye[0] + left_eye[0]) // 2
            glasses_width = int(face_width * 2.0)
            glasses_height = int(glasses_img.shape[0] * (glasses_width / glasses_img.shape[1]))
            gx = center_x - glasses_width // 2 - 10
            gy = int((left_eye[1] + right_eye[1]) / 2) - glasses_height // 2 + 15
            frame = overlay_image(frame, glasses_img, gx, gy, (glasses_width, glasses_height))

            # === Mustache positioning ===
            mustache_width = glasses_width
            mustache_height = mustache_width // 3
            mx = nose_tip[0] - mustache_width // 2
            my = upper_lip[1] - 35
            frame = overlay_image(frame, mustache_img, mx, my, (mustache_width, mustache_height))

    cv2.imshow("Thug Life Filter - Multi Face", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
