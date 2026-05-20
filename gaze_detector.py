import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Index landmark iris dari MediaPipe
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE   = [33, 133]   # sudut kiri & kanan mata kiri
RIGHT_EYE  = [362, 263]  # sudut kiri & kanan mata kanan

def get_iris_position(landmarks, iris_idx, eye_idx, w, h):
    """Hitung posisi relatif iris dalam bounding box mata (0.0 - 1.0)"""
    iris_pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in iris_idx])
    iris_center = iris_pts.mean(axis=0)

    eye_left  = np.array([landmarks[eye_idx[0]].x * w, landmarks[eye_idx[0]].y * h])
    eye_right = np.array([landmarks[eye_idx[1]].x * w, landmarks[eye_idx[1]].y * h])

    eye_width = np.linalg.norm(eye_right - eye_left)
    ratio_x = (iris_center[0] - eye_left[0]) / eye_width  # 0 = kiri, 1 = kanan

    return iris_center, ratio_x

def classify_gaze(ratio_left, ratio_right, threshold=0.15):
    """
    Rata-rata posisi iris kedua mata.
    Kalau mendekati 0.5 → menatap lurus ke kamera.
    """
    avg = (ratio_left + ratio_right) / 2
    deviation = abs(avg - 0.5)

    if deviation < threshold:
        return "LOOKING AT CAMERA", (0, 200, 100)
    elif avg < 0.5 - threshold:
        return "Looking LEFT", (0, 165, 255)
    else:
        return "Looking RIGHT", (0, 165, 255)

def run(source=0):
    """
    source=0       → webcam laptop
    source='x.mp4' → file video
    """
    cap = cv2.VideoCapture(source)
    attention_count = 0
    total_frames = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,   # wajib untuk iris
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb) 
            total_frames += 1
            looking_count = 0

            if results.multi_face_landmarks:
                for face_lm in results.multi_face_landmarks:
                    lm = face_lm.landmark

                    # Hitung posisi iris
                    lc, ratio_l = get_iris_position(lm, LEFT_IRIS,  LEFT_EYE,  w, h)
                    rc, ratio_r = get_iris_position(lm, RIGHT_IRIS, RIGHT_EYE, w, h)

                    label, color = classify_gaze(ratio_l, ratio_r)

                    if "LOOKING AT CAMERA" in label:
                        looking_count += 1
                        attention_count += 1

                    # Gambar titik iris
                    cv2.circle(frame, tuple(lc.astype(int)), 3, color, -1)
                    cv2.circle(frame, tuple(rc.astype(int)), 3, color, -1)

                    # Label per wajah
                    x0 = int(lm[10].x * w)
                    y0 = int(lm[10].y * h) - 20
                    cv2.putText(frame, label, (x0 - 60, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Statistik attention rate
            rate = (attention_count / max(total_frames, 1)) * 100
            cv2.putText(frame, f"Attention rate: {rate:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Looking now: {looking_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

            cv2.imshow("Eye Gaze Detector (tekan Q untuk keluar)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSelesai! Attention rate akhir: {rate:.1f}%")

if __name__ == "__main__":
    run(source=0)  # ganti ke 'video.mp4' kalau pakai file
    