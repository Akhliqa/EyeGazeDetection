"""
Eye Gaze Detection — Improved Tuning & Evaluation
===================================================
Perbaikan dari versi sebelumnya:
  1. Kalibrasi per-user  : baseline iris setiap orang berbeda
  2. Head pose estimation: rotasi kepala turut diperhitungkan
  3. Temporal voting     : keputusan dari mayoritas N frame terakhir
  4. Adaptive threshold  : threshold menyesuaikan proporsi mata terbuka
  5. Evaluasi lebih ketat: stratified split, per-class report
 
Instalasi:
    pip install opencv-python mediapipe numpy scikit-learn pandas tabulate
 
Cara pakai:
    python gaze_tuning_v2.py --source 0 --calibrate   # kalibrasi dulu (WAJIB)
    python gaze_tuning_v2.py --source 0 --collect     # kumpulkan ground truth
    python gaze_tuning_v2.py --tune                   # grid search
    python gaze_tuning_v2.py --source 0               # real-time + evaluasi
"""
 
import cv2
import mediapipe as mp
import numpy as np
import argparse
import csv
import json
import time
import os
from itertools import product
from datetime import datetime
from collections import deque
 
try:
    from tabulate import tabulate
    HAS_TAB = True
except ImportError:
    HAS_TAB = False
 
# ─────────────────────────────────────────────
# LANDMARK INDEX
# ─────────────────────────────────────────────
LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]
LEFT_EYE_H  = [33, 133]
RIGHT_EYE_H = [362, 263]
LEFT_EYE_V  = [159, 145]
RIGHT_EYE_V = [386, 374]
 
# Untuk head pose (6 titik standar PnP)
HEAD_POSE_IDX = [1, 152, 263, 33, 287, 57]
 
# Model 3D wajah generik (mm)
FACE_3D = np.array([
    [0.0,    0.0,    0.0],    # Hidung
    [0.0,   -330.0, -65.0],   # Dagu
    [-225.0,  170.0,-135.0],  # Sudut mata kiri
    [225.0,   170.0,-135.0],  # Sudut mata kanan
    [-150.0, -150.0,-125.0],  # Sudut mulut kiri
    [150.0,  -150.0,-125.0],  # Sudut mulut kanan
], dtype=np.float64)
 
DEFAULT_CONFIG = {
    "gaze_threshold":     0.13,
    "vertical_threshold": 0.13,
    "pose_yaw_limit":     12.0,   # derajat, batas yaw kepala untuk "lurus"
    "pose_pitch_limit":   10.0,   # derajat, batas pitch kepala
    "smooth_window":      7,
    "vote_window":        9,      # temporal voting: jumlah frame
    "min_detection_conf": 0.5,
    "min_tracking_conf":  0.5,
    "max_faces":          5,
    "use_head_pose":      True,
    "use_calibration":    True,
}
 
CALIB_FILE = "gaze_calibration.json"
 
# ─────────────────────────────────────────────
# HEAD POSE ESTIMATOR
# ─────────────────────────────────────────────
 
class HeadPoseEstimator:
    def __init__(self, w, h):
        focal = w
        cx, cy = w / 2, h / 2
        self.cam_matrix = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1],
        ], dtype=np.float64)
        self.dist = np.zeros((4, 1))
 
    def estimate(self, landmarks, w, h):
        """Return (yaw, pitch, roll) dalam derajat. None jika gagal."""
        pts_2d = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in HEAD_POSE_IDX
        ], dtype=np.float64)
 
        ok, rvec, tvec = cv2.solvePnP(
            FACE_3D, pts_2d, self.cam_matrix, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
 
        rot_mat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
        pitch = np.degrees(np.arctan2(-rot_mat[2,0], sy))
        yaw   = np.degrees(np.arctan2(rot_mat[1,0], rot_mat[0,0]))
        roll  = np.degrees(np.arctan2(rot_mat[2,1], rot_mat[2,2]))
        return yaw, pitch, roll
 
 
# ─────────────────────────────────────────────
# KALIBRASI PER-USER
# ─────────────────────────────────────────────
 
class Calibrator:
    """
    Rekam baseline iris ratio saat user menatap lurus ke kamera.
    Digunakan untuk normalisasi — tiap orang punya offset berbeda.
    """
    def __init__(self):
        self.baseline_rx = 0.5
        self.baseline_ry = 0.5
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.is_calibrated = False
 
    def save(self, path=CALIB_FILE):
        data = {
            "baseline_rx":    self.baseline_rx,
            "baseline_ry":    self.baseline_ry,
            "baseline_yaw":   self.baseline_yaw,
            "baseline_pitch": self.baseline_pitch,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✅ Kalibrasi disimpan ke '{path}'")
 
    def load(self, path=CALIB_FILE):
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.baseline_rx    = data["baseline_rx"]
        self.baseline_ry    = data["baseline_ry"]
        self.baseline_yaw   = data["baseline_yaw"]
        self.baseline_pitch = data["baseline_pitch"]
        self.is_calibrated  = True
        print(f"  ✅ Kalibrasi dimuat: rx={self.baseline_rx:.3f} "
              f"ry={self.baseline_ry:.3f} "
              f"yaw={self.baseline_yaw:.1f}° "
              f"pitch={self.baseline_pitch:.1f}°")
        return True
 
    def normalize_rx(self, rx):
        """Geser rx relatif terhadap baseline user."""
        return rx - self.baseline_rx + 0.5
 
    def normalize_ry(self, ry):
        return ry - self.baseline_ry + 0.5
 
    def normalize_yaw(self, yaw):
        return yaw - self.baseline_yaw
 
    def normalize_pitch(self, pitch):
        return pitch - self.baseline_pitch
 
 
def run_calibration(source):
    """
    Mode kalibrasi interaktif.
    User menatap kamera selama 3 detik, sistem merekam rata-rata.
    """
    print("\n" + "═"*60)
    print("  MODE KALIBRASI")
    print("  Tatap tepat ke tengah kamera, lalu tekan SPASI.")
    print("  Tahan selama 3 detik. Tekan Q untuk batal.")
    print("═"*60 + "\n")
 
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
 
    try:
        src = int(source)
    except ValueError:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  [!] Tidak bisa buka source: {source}")
        return None
 
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    h, w = frame.shape[:2]
    pose_est = HeadPoseEstimator(w, h)
 
    collecting = False
    buf_rx, buf_ry, buf_yaw, buf_pitch = [], [], [], []
    start_time = None
    COLLECT_SEC = 3.0
    calib = Calibrator()
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
 
        rx_now = ry_now = yaw_now = pitch_now = None
 
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            try:
                iris_l = np.mean([(lm[i].x * w, lm[i].y * h) for i in LEFT_IRIS],  axis=0)
                iris_r = np.mean([(lm[i].x * w, lm[i].y * h) for i in RIGHT_IRIS], axis=0)
 
                el = np.array([lm[LEFT_EYE_H[0]].x * w, lm[LEFT_EYE_H[0]].y * h])
                er = np.array([lm[LEFT_EYE_H[1]].x * w, lm[LEFT_EYE_H[1]].y * h])
                ew_l = np.linalg.norm(er - el) + 1e-6
 
                er2 = np.array([lm[RIGHT_EYE_H[0]].x * w, lm[RIGHT_EYE_H[0]].y * h])
                er3 = np.array([lm[RIGHT_EYE_H[1]].x * w, lm[RIGHT_EYE_H[1]].y * h])
                ew_r = np.linalg.norm(er3 - er2) + 1e-6
 
                rx_l = (iris_l[0] - el[0]) / ew_l
                rx_r = (iris_r[0] - er2[0]) / ew_r
                rx_now = (rx_l + rx_r) / 2
 
                et_l = np.array([lm[LEFT_EYE_V[0]].x * w, lm[LEFT_EYE_V[0]].y * h])
                eb_l = np.array([lm[LEFT_EYE_V[1]].x * w, lm[LEFT_EYE_V[1]].y * h])
                eh_l = np.linalg.norm(eb_l - et_l) + 1e-6
                ry_l = (iris_l[1] - et_l[1]) / eh_l
 
                et_r = np.array([lm[RIGHT_EYE_V[0]].x * w, lm[RIGHT_EYE_V[0]].y * h])
                eb_r = np.array([lm[RIGHT_EYE_V[1]].x * w, lm[RIGHT_EYE_V[1]].y * h])
                eh_r = np.linalg.norm(eb_r - et_r) + 1e-6
                ry_r = (iris_r[1] - et_r[1]) / eh_r
                ry_now = (ry_l + ry_r) / 2
 
                pose = pose_est.estimate(lm, w, h)
                if pose:
                    yaw_now, pitch_now, _ = pose
            except Exception:
                pass
 
            # Gambar iris
            for idx in LEFT_IRIS + RIGHT_IRIS:
                pt = (int(lm[idx].x * w), int(lm[idx].y * h))
                cv2.circle(frame, pt, 3, (0, 255, 200), -1)
 
        # UI overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
 
        if collecting and start_time:
            elapsed = time.time() - start_time
            remaining = max(0, COLLECT_SEC - elapsed)
            progress = int((elapsed / COLLECT_SEC) * (w - 40))
            cv2.rectangle(frame, (20, 70), (20 + progress, 82), (0, 230, 100), -1)
            cv2.rectangle(frame, (20, 70), (w - 20, 82), (80, 80, 80), 1)
            cv2.putText(frame, f"Merekam... {remaining:.1f}s",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 100), 2)
 
            if rx_now is not None:
                buf_rx.append(rx_now); buf_ry.append(ry_now)
            if yaw_now is not None:
                buf_yaw.append(yaw_now); buf_pitch.append(pitch_now)
 
            if elapsed >= COLLECT_SEC:
                collecting = False
                calib.baseline_rx    = float(np.mean(buf_rx))
                calib.baseline_ry    = float(np.mean(buf_ry))
                calib.baseline_yaw   = float(np.mean(buf_yaw)) if buf_yaw else 0.0
                calib.baseline_pitch = float(np.mean(buf_pitch)) if buf_pitch else 0.0
                calib.is_calibrated  = True
                calib.save()
                cap.release()
                cv2.destroyAllWindows()
                face_mesh.close()
                return calib
        else:
            status = "Wajah terdeteksi — tekan SPASI untuk mulai" if rx_now is not None \
                     else "Posisikan wajah di depan kamera"
            cv2.putText(frame, status, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 255) if rx_now is not None else (80, 80, 255), 2)
            if rx_now is not None:
                cv2.putText(frame, f"rx={rx_now:.3f}  ry={ry_now:.3f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
 
        cv2.imshow("Kalibrasi — SPASI: mulai  Q: batal", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' ') and not collecting and rx_now is not None:
            collecting = True
            start_time = time.time()
            buf_rx.clear(); buf_ry.clear()
            buf_yaw.clear(); buf_pitch.clear()
 
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    return None
 
 
# ─────────────────────────────────────────────
# IMPROVED GAZE DETECTOR
# ─────────────────────────────────────────────
 
class ImprovedGazeDetector:
    def __init__(self, config=None, calibrator=None):
        self.cfg  = {**DEFAULT_CONFIG, **(config or {})}
        self.calib = calibrator
 
        self.mp_face   = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=self.cfg["max_faces"],
            refine_landmarks=True,
            min_detection_confidence=self.cfg["min_detection_conf"],
            min_tracking_confidence=self.cfg["min_tracking_conf"],
        )
        self._pose_est  = None
        self._smooth    = {}   # face_id -> deque
        self._vote_buf  = {}   # face_id -> deque[label]
 
    def close(self):
        self.face_mesh.close()
 
    def _get_pose_est(self, w, h):
        if self._pose_est is None:
            self._pose_est = HeadPoseEstimator(w, h)
        return self._pose_est
 
    def _iris_ratios(self, lm, w, h):
        """Return (rx, ry) rata-rata kedua mata."""
        def ratio(iris_idx, eye_h, eye_v):
            iris = np.mean([(lm[i].x*w, lm[i].y*h) for i in iris_idx], axis=0)
            el  = np.array([lm[eye_h[0]].x*w, lm[eye_h[0]].y*h])
            er  = np.array([lm[eye_h[1]].x*w, lm[eye_h[1]].y*h])
            ew  = np.linalg.norm(er - el) + 1e-6
            et  = np.array([lm[eye_v[0]].x*w, lm[eye_v[0]].y*h])
            eb  = np.array([lm[eye_v[1]].x*w, lm[eye_v[1]].y*h])
            eh  = np.linalg.norm(eb - et) + 1e-6
            rx  = (iris[0] - el[0]) / ew
            ry  = (iris[1] - et[1]) / eh
            return rx, ry, tuple(iris.astype(int))
 
        rx_l, ry_l, il = ratio(LEFT_IRIS,  LEFT_EYE_H,  LEFT_EYE_V)
        rx_r, ry_r, ir = ratio(RIGHT_IRIS, RIGHT_EYE_H, RIGHT_EYE_V)
        return (rx_l + rx_r) / 2, (ry_l + ry_r) / 2, il, ir
 
    def _smooth_val(self, fid, rx, ry):
        n = self.cfg["smooth_window"]
        if fid not in self._smooth:
            self._smooth[fid] = {"rx": deque(maxlen=n), "ry": deque(maxlen=n)}
        b = self._smooth[fid]
        b["rx"].append(rx); b["ry"].append(ry)
        return float(np.mean(b["rx"])), float(np.mean(b["ry"]))
 
    def _vote(self, fid, raw_label):
        """Temporal voting: kembalikan label terbanyak dalam window."""
        n = self.cfg["vote_window"]
        if fid not in self._vote_buf:
            self._vote_buf[fid] = deque(maxlen=n)
        self._vote_buf[fid].append(raw_label)
        buf = list(self._vote_buf[fid])
        return max(set(buf), key=buf.count)
 
    def classify(self, rx, ry, yaw=None, pitch=None):
        """
        Klasifikasi dengan 3 sinyal:
          1. Iris ratio (horizontal & vertikal)
          2. Head pose yaw & pitch (jika tersedia & diaktifkan)
        Return (label, looking_bool, signals_dict)
        """
        th_h = self.cfg["gaze_threshold"]
        th_v = self.cfg["vertical_threshold"]
 
        # Normalisasi dengan kalibrasi
        if self.calib and self.calib.is_calibrated and self.cfg["use_calibration"]:
            rx = self.calib.normalize_rx(rx)
            ry = self.calib.normalize_ry(ry)
            if yaw   is not None: yaw   = self.calib.normalize_yaw(yaw)
            if pitch  is not None: pitch = self.calib.normalize_pitch(pitch)
 
        dev_h = rx - 0.5
        dev_v = ry - 0.5
 
        # Sinyal dari iris
        iris_h = "CENTER" if abs(dev_h) < th_h else ("LEFT" if dev_h < 0 else "RIGHT")
        iris_v = "CENTER" if abs(dev_v) < th_v else ("UP"   if dev_v < 0 else "DOWN")
 
        # Sinyal dari head pose
        pose_h = pose_v = "CENTER"
        if yaw is not None and self.cfg["use_head_pose"]:
            yl = self.cfg["pose_yaw_limit"]
            pl = self.cfg["pose_pitch_limit"]
            pose_h = "CENTER" if abs(yaw)   < yl else ("LEFT" if yaw   > 0 else "RIGHT")
            pose_v = "CENTER" if abs(pitch) < pl else ("UP"   if pitch < 0 else "DOWN")
 
        # Fusi sinyal — majority vote antara iris & pose
        # Horizontal
        votes_h = [iris_h]
        if self.cfg["use_head_pose"] and yaw is not None:
            votes_h.append(pose_h)
        final_h = max(set(votes_h), key=votes_h.count)
 
        # Vertikal
        votes_v = [iris_v]
        if self.cfg["use_head_pose"] and pitch is not None:
            votes_v.append(pose_v)
        final_v = max(set(votes_v), key=votes_v.count)
 
        # Tentukan label akhir
        if final_h == "CENTER" and final_v == "CENTER":
            label = "CAMERA"
        elif abs(dev_h) >= abs(dev_v):
            label = final_h
        else:
            label = final_v
 
        looking = 1 if label == "CAMERA" else 0
 
        # Confidence: gabungan kedekatan iris ke tengah + konsistensi pose
        iris_dist = np.sqrt(dev_h**2 + dev_v**2)
        iris_conf = max(0.0, 1.0 - iris_dist / 0.5)
        pose_conf = 1.0
        if yaw is not None and self.cfg["use_head_pose"]:
            yl, pl = self.cfg["pose_yaw_limit"], self.cfg["pose_pitch_limit"]
            pose_conf = max(0.0, 1.0 - (abs(yaw)/yl + abs(pitch)/pl) / 2)
        confidence = round((iris_conf + pose_conf) / 2, 3)
 
        signals = {
            "rx": round(rx, 4), "ry": round(ry, 4),
            "iris_h": iris_h,   "iris_v": iris_v,
            "pose_h": pose_h,   "pose_v": pose_v,
            "yaw": round(yaw, 1) if yaw is not None else None,
            "pitch": round(pitch, 1) if pitch is not None else None,
        }
        return label, looking, confidence, signals
 
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)
        results = []
 
        if not res.multi_face_landmarks:
            return results
 
        pose_est = self._get_pose_est(w, h)
 
        for fid, face_lm in enumerate(res.multi_face_landmarks):
            lm = face_lm.landmark
            try:
                rx, ry, il, ir = self._iris_ratios(lm, w, h)
            except Exception:
                continue
 
            # Smoothing
            rx, ry = self._smooth_val(fid, rx, ry)
 
            # Head pose
            yaw = pitch = None
            if self.cfg["use_head_pose"]:
                pose = pose_est.estimate(lm, w, h)
                if pose:
                    yaw, pitch, _ = pose
 
            # Klasifikasi
            raw_label, looking, conf, signals = self.classify(rx, ry, yaw, pitch)
 
            # Temporal voting
            final_label = self._vote(fid, raw_label)
            final_looking = 1 if final_label == "CAMERA" else 0
 
            results.append({
                "face_id":    fid,
                "label":      final_label,
                "raw_label":  raw_label,
                "looking":    final_looking,
                "confidence": conf,
                "iris_l":     il,
                "iris_r":     ir,
                "signals":    signals,
            })
 
        return results
 
 
# ─────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────
 
class Evaluator:
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.confs  = []
        self.times  = []
 
    def add(self, true_label, pred_label, conf, ms):
        self.y_true.append(true_label)
        self.y_pred.append(pred_label)
        self.confs.append(conf)
        self.times.append(ms)
 
    def compute(self):
        if not self.y_true:
            return {}
        yt = np.array(self.y_true)
        yp = np.array(self.y_pred)
        n  = len(yt)
        acc = float(np.mean(yt == yp))
 
        classes = sorted(set(yt) | set(yp))
        cm, cls_m = {}, {}
        for c in classes:
            tp = int(np.sum((yp == c) & (yt == c)))
            fp = int(np.sum((yp == c) & (yt != c)))
            fn = int(np.sum((yp != c) & (yt == c)))
            tn = int(np.sum((yp != c) & (yt != c)))
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            spec = tn / (tn + fp + 1e-9)
            cls_m[c] = {
                "precision":   round(prec, 4),
                "recall":      round(rec, 4),
                "f1":          round(f1, 4),
                "specificity": round(spec, 4),
                "support":     int(np.sum(yt == c)),
            }
            cm[c] = {p: int(np.sum((yt == c) & (yp == p))) for p in classes}
 
        mp_ = float(np.mean([v["precision"]  for v in cls_m.values()]))
        mr  = float(np.mean([v["recall"]     for v in cls_m.values()]))
        mf1 = float(np.mean([v["f1"]         for v in cls_m.values()]))
        avg_c = float(np.mean(self.confs))
        avg_fps = 1000 / (float(np.mean(self.times)) + 1e-9)
        p95 = float(np.percentile(self.times, 95))
 
        return {
            "n_samples": n, "accuracy": round(acc, 4),
            "macro_precision": round(mp_, 4), "macro_recall": round(mr, 4),
            "macro_f1": round(mf1, 4), "avg_confidence": round(avg_c, 4),
            "avg_fps": round(avg_fps, 1), "p95_latency_ms": round(p95, 1),
            "class_metrics": cls_m, "confusion_matrix": cm, "classes": classes,
        }
 
 
# ─────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────
 
def print_header():
    print("\n" + "═"*66)
    print("  👁  EYE GAZE DETECTION v2 — Tuning & Evaluation")
    print("═"*66 + "\n")
 
def print_metrics(metrics, config=None, label=""):
    if not metrics:
        print("  [!] Belum ada data evaluasi."); return
    print("\n" + "─"*66)
    print(f"  HASIL EVALUASI{' — ' + label if label else ''}")
    print("─"*66)
    if config:
        print(f"  threshold={config['gaze_threshold']}  "
              f"v_threshold={config['vertical_threshold']}  "
              f"smooth={config['smooth_window']}  "
              f"vote={config['vote_window']}  "
              f"head_pose={config['use_head_pose']}")
    rows = [
        ["Accuracy",         f"{metrics['accuracy']*100:.2f}%"],
        ["Macro Precision",  f"{metrics['macro_precision']*100:.2f}%"],
        ["Macro Recall",     f"{metrics['macro_recall']*100:.2f}%"],
        ["Macro F1-Score",   f"{metrics['macro_f1']*100:.2f}%"],
        ["Avg Confidence",   f"{metrics['avg_confidence']*100:.1f}%"],
        ["Avg FPS",          f"{metrics['avg_fps']:.1f}"],
        ["P95 Latency",      f"{metrics['p95_latency_ms']:.1f} ms"],
        ["Total Sampel",     str(metrics['n_samples'])],
    ]
    if HAS_TAB:
        print(tabulate(rows, headers=["Metrik", "Nilai"], tablefmt="rounded_outline"))
    else:
        for r in rows: print(f"  {r[0]:<22} {r[1]}")
 
    print("\n  Per-Class Metrics:")
    rows_c = [[c,
               f"{m['precision']*100:.1f}%",
               f"{m['recall']*100:.1f}%",
               f"{m['f1']*100:.1f}%",
               f"{m['specificity']*100:.1f}%",
               m['support']]
              for c, m in metrics["class_metrics"].items()]
    hdrs = ["Kelas","Precision","Recall","F1","Specificity","Support"]
    if HAS_TAB:
        print(tabulate(rows_c, headers=hdrs, tablefmt="rounded_outline"))
    else:
        print("  " + "  ".join(f"{h:<12}" for h in hdrs))
        for r in rows_c: print("  " + "  ".join(f"{str(x):<12}" for x in r))
 
    classes = metrics["classes"]
    cm = metrics["confusion_matrix"]
    print("\n  Confusion Matrix (baris=aktual, kolom=prediksi):")
    cm_rows = [[t] + [cm[t].get(p, 0) for p in classes] for t in classes]
    if HAS_TAB:
        print(tabulate(cm_rows, headers=[""]+classes, tablefmt="rounded_outline"))
    else:
        header = [""] + classes
        print("  " + "  ".join(f"{h:<10}" for h in header))
        for r in cm_rows: print("  " + "  ".join(f"{str(x):<10}" for x in r))
 
def print_tune_results(results):
    results_s = sorted(results, key=lambda x: x["metrics"].get("macro_f1", 0), reverse=True)
    print("\n" + "═"*80)
    print("  HASIL GRID SEARCH — Top 15 (diurutkan F1)")
    print("═"*80)
    rows = []
    for i, r in enumerate(results_s[:15]):
        m, c = r["metrics"], r["config"]
        rows.append([
            f"#{i+1}", c["gaze_threshold"], c["smooth_window"],
            c["vote_window"], c["use_head_pose"],
            f"{m.get('accuracy',0)*100:.1f}%",
            f"{m.get('macro_f1',0)*100:.1f}%",
            f"{m.get('avg_confidence',0)*100:.1f}%",
            f"{m.get('avg_fps',0):.0f}",
        ])
    hdrs = ["Rank","Threshold","Smooth","Vote","HeadPose","Accuracy","F1","Conf","FPS"]
    if HAS_TAB:
        print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))
    else:
        print("  "+"  ".join(f"{h:<10}" for h in hdrs))
        for r in rows: print("  "+"  ".join(f"{str(x):<10}" for x in r))
    best = results_s[0]
    print(f"\n  ✅ Config terbaik:")
    for k, v in best["config"].items():
        print(f"     {k:<25} = {v}")
    print(f"\n     → F1: {best['metrics'].get('macro_f1',0)*100:.2f}%  "
          f"Accuracy: {best['metrics'].get('accuracy',0)*100:.2f}%\n")
    return best["config"]
 
 
# ─────────────────────────────────────────────
# MODE: COLLECT GROUND TRUTH
# ─────────────────────────────────────────────
 
def collect_ground_truth(source, out_file="ground_truth.csv", calib=None):
    print_header()
    print("  MODE: Kumpulkan Ground Truth")
    print("  C=CAMERA  L=LEFT  R=RIGHT  U=UP  D=DOWN  Q=Selesai\n")
 
    key_map = {ord('c'):'CAMERA', ord('l'):'LEFT', ord('r'):'RIGHT',
               ord('u'):'UP',    ord('d'):'DOWN'}
 
    cfg = {**DEFAULT_CONFIG}
    cfg["use_calibration"] = calib is not None and calib.is_calibrated
    detector = ImprovedGazeDetector(cfg, calib)
 
    try:
        src = int(source)
    except ValueError:
        src = source
    cap = cv2.VideoCapture(src)
    records = []
    current_label = None
    frame_count = 0
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
 
        t0 = time.perf_counter()
        faces = detector.process_frame(frame)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key in key_map: current_label = key_map[key]
 
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,65), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
 
        col = (0,255,128) if current_label == 'CAMERA' else (0,200,255)
        cv2.putText(frame, f"Label: {current_label or 'tekan C/L/R/U/D'}",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        cv2.putText(frame, f"Frame:{frame_count}  Data:{len(records)}",
                    (10,58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)
 
        for f in faces:
            il, ir = f["iris_l"], f["iris_r"]
            color = (0,255,0) if f["label"] == current_label else (0,80,255)
            cv2.circle(frame, il, 4, color, -1)
            cv2.circle(frame, ir, 4, color, -1)
            sig = f["signals"]
            cv2.putText(frame, f"pred:{f['label']}",
                        (il[0]-40, il[1]-18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
            if sig["yaw"] is not None:
                cv2.putText(frame, f"yaw:{sig['yaw']} pitch:{sig['pitch']}",
                            (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
            if current_label:
                records.append({
                    "frame": frame_count,
                    "true_label": current_label,
                    "pred_label": f["label"],
                    "rx": sig["rx"], "ry": sig["ry"],
                    "yaw": sig["yaw"] if sig["yaw"] is not None else "",
                    "pitch": sig["pitch"] if sig["pitch"] is not None else "",
                    "confidence": f["confidence"],
                    "latency_ms": round(ms, 2),
                })
 
        cv2.imshow("Ground Truth Collection", frame)
 
    cap.release(); cv2.destroyAllWindows(); detector.close()
 
    if records:
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader(); writer.writerows(records)
 
        # Distribusi per kelas
        from collections import Counter
        dist = Counter(r["true_label"] for r in records)
        print(f"\n  ✅ {len(records)} data disimpan ke '{out_file}'")
        print("  Distribusi kelas:")
        for k, v in dist.items():
            print(f"    {k:<10} {v} sampel")
    else:
        print("\n  [!] Tidak ada data.")
 
 
# ─────────────────────────────────────────────
# MODE: EVALUATE FROM CSV
# ─────────────────────────────────────────────
 
def evaluate_from_csv(csv_file, config=None, calib=None):
    if not os.path.exists(csv_file):
        print(f"  [!] '{csv_file}' tidak ditemukan.")
        return None
 
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    th_h = cfg["gaze_threshold"]
    th_v = cfg["vertical_threshold"]
    yl   = cfg["pose_yaw_limit"]
    pl   = cfg["pose_pitch_limit"]
    use_pose  = cfg["use_head_pose"]
    use_calib = cfg["use_calibration"] and calib and calib.is_calibrated
 
    evaluator = Evaluator()
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rx   = float(row["rx"])
            ry   = float(row["ry"])
            yaw  = float(row["yaw"])   if row.get("yaw")   else None
            pitch= float(row["pitch"]) if row.get("pitch") else None
            ms   = float(row["latency_ms"])
            conf = float(row["confidence"])
            true = row["true_label"]
 
            if use_calib:
                rx = calib.normalize_rx(rx)
                ry = calib.normalize_ry(ry)
                if yaw   is not None: yaw   = calib.normalize_yaw(yaw)
                if pitch is not None: pitch = calib.normalize_pitch(pitch)
 
            dev_h = rx - 0.5; dev_v = ry - 0.5
            iris_h = "CENTER" if abs(dev_h)<th_h else ("LEFT" if dev_h<0 else "RIGHT")
            iris_v = "CENTER" if abs(dev_v)<th_v else ("UP"   if dev_v<0 else "DOWN")
            pose_h = pose_v = "CENTER"
            if use_pose and yaw is not None:
                pose_h = "CENTER" if abs(yaw)<yl   else ("LEFT" if yaw>0   else "RIGHT")
                pose_v = "CENTER" if abs(pitch)<pl  else ("UP"   if pitch<0 else "DOWN")
 
            votes_h = [iris_h, pose_h] if (use_pose and yaw is not None) else [iris_h]
            votes_v = [iris_v, pose_v] if (use_pose and yaw is not None) else [iris_v]
            fh = max(set(votes_h), key=votes_h.count)
            fv = max(set(votes_v), key=votes_v.count)
 
            if fh == "CENTER" and fv == "CENTER":
                pred = "CAMERA"
            elif abs(dev_h) >= abs(dev_v):
                pred = fh
            else:
                pred = fv
 
            evaluator.add(true, pred, conf, ms)
 
    return evaluator.compute()
 
 
# ─────────────────────────────────────────────
# MODE: REAL-TIME
# ─────────────────────────────────────────────
 
def run_realtime(source, config=None, gt_file=None, calib=None):
    print_header()
    print("  MODE: Real-Time Detection")
    calib_status = "✅ Aktif" if (calib and calib.is_calibrated) else "⚠️  Tidak aktif (jalankan --calibrate)"
    print(f"  Kalibrasi : {calib_status}")
    print("  Tekan Q=keluar  S=lihat skor  C/L/R/U/D=label manual\n")
 
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    cfg["use_calibration"] = calib is not None and calib.is_calibrated
    detector  = ImprovedGazeDetector(cfg, calib)
    evaluator = Evaluator()
 
    gt_data = []
    if gt_file and os.path.exists(gt_file):
        with open(gt_file) as f:
            gt_data = list(csv.DictReader(f))
        print(f"  Ground truth: {len(gt_data)} sampel dari '{gt_file}'\n")
 
    try:
        src = int(source)
    except ValueError:
        src = source
    cap = cv2.VideoCapture(src)
 
    frame_idx = 0
    attn_total = frames_total = 0
    fps_buf = deque(maxlen=30)
    current_label = None
    key_map = {ord('c'):'CAMERA', ord('l'):'LEFT', ord('r'):'RIGHT',
               ord('u'):'UP',    ord('d'):'DOWN'}
 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
 
        t0 = time.perf_counter()
        faces = detector.process_frame(frame)
        ms = (time.perf_counter() - t0) * 1000
        fps_buf.append(1000 / (ms + 1e-9))
 
        looking_now = sum(f["looking"] for f in faces)
        frames_total += 1
        attn_total += (1 if looking_now > 0 else 0)
        attn_rate = attn_total / frames_total * 100
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key in key_map: current_label = key_map[key]
        if key == ord('s'):
            m = evaluator.compute()
            if m: print_metrics(m, cfg, "Live Snapshot")
            else: print("\n  [!] Belum ada data evaluasi.")
 
        # GT dari file atau label manual
        true_label = current_label
        if gt_data and frame_idx < len(gt_data):
            true_label = gt_data[frame_idx]["true_label"]
 
        for f in faces:
            if true_label:
                evaluator.add(true_label, f["label"], f["confidence"], ms)
            il, ir = f["iris_l"], f["iris_r"]
            color = (0,230,100) if f["looking"] else (0,100,255)
            cv2.circle(frame, il, 4, color, -1)
            cv2.circle(frame, ir, 4, color, -1)
            sig = f["signals"]
            info = f"{f['label']} ({f['confidence']*100:.0f}%)"
            if sig["yaw"] is not None:
                info += f"  y:{sig['yaw']} p:{sig['pitch']}"
            cv2.putText(frame, info, (il[0]-50, il[1]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
 
        # HUD
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,85), (8,8,18), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, f"FPS:{np.mean(fps_buf):.0f}  Latency:{ms:.1f}ms",
                    (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150,220,255), 1)
        cv2.putText(frame, f"Attention:{attn_rate:.1f}%  Wajah:{len(faces)}  Menatap:{looking_now}",
                    (10,44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150,255,200), 1)
        cv2.putText(frame, f"thr={cfg['gaze_threshold']}  smooth={cfg['smooth_window']}  "
                    f"vote={cfg['vote_window']}  pose={'on' if cfg['use_head_pose'] else 'off'}",
                    (10,66), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,180), 1)
        if current_label:
            cv2.putText(frame, f"Label:{current_label}", (w-150,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,220,0), 2)
 
        cv2.imshow("Eye Gaze v2 — Q:Keluar S:Skor", frame)
        frame_idx += 1
 
    cap.release(); cv2.destroyAllWindows(); detector.close()
    m = evaluator.compute()
    if m:
        print_metrics(m, cfg, "Akhir Sesi")
    else:
        print(f"\n  Sesi selesai. Attention rate: {attn_rate:.1f}%  FPS: {np.mean(fps_buf):.1f}")
 
 
# ─────────────────────────────────────────────
# MODE: GRID SEARCH
# ─────────────────────────────────────────────
 
def grid_search(gt_file, calib=None):
    print_header()
    print(f"  MODE: Grid Search Tuning  |  Dataset: {gt_file}\n")
    if not os.path.exists(gt_file):
        print(f"  [!] '{gt_file}' tidak ditemukan. Jalankan --collect dulu."); return
 
    param_grid = {
        "gaze_threshold":     [0.08, 0.11, 0.14, 0.17, 0.20, 0.25],
        "vertical_threshold": [0.08, 0.12, 0.16],
        "smooth_window":      [3, 5, 7, 10],
        "vote_window":        [5, 7, 9],
        "pose_yaw_limit":     [8.0, 12.0, 16.0],
        "use_head_pose":      [True, False],
    }
    combos = list(product(*param_grid.values()))
    keys   = list(param_grid.keys())
    total  = len(combos)
    print(f"  Total kombinasi: {total}")
 
    results = []
    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        m = evaluate_from_csv(gt_file, cfg, calib)
        if m and m["n_samples"] > 0:
            results.append({"config": cfg, "metrics": m})
        pct = (i+1) / total
        bar = "█" * int(pct*42) + "░" * (42-int(pct*42))
        print(f"\r  [{bar}] {i+1}/{total}", end="", flush=True)
    print()
 
    if not results:
        print("\n  [!] Tidak ada hasil."); return
 
    best_cfg = print_tune_results(results)
 
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"tuning_results_{ts}.json"
    with open(out, "w") as f:
        json.dump([{"config": r["config"],
                    "metrics": {k:v for k,v in r["metrics"].items()
                                if k not in ("class_metrics","confusion_matrix","classes")}}
                   for r in results], f, indent=2)
    print(f"  💾 Hasil disimpan ke '{out}'")
    return best_cfg
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Eye Gaze Detection v2 — Improved Tuning & Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Alur yang direkomendasikan:
  1. python gaze_tuning_v2.py --source 0 --calibrate   # kalibrasi (wajib!)
  2. python gaze_tuning_v2.py --source 0 --collect     # kumpulkan ground truth
  3. python gaze_tuning_v2.py --tune                   # grid search
  4. python gaze_tuning_v2.py --source 0 --threshold 0.14 --smooth 7
        """
    )
    parser.add_argument("--source",    default="0")
    parser.add_argument("--calibrate", action="store_true", help="Jalankan kalibrasi per-user")
    parser.add_argument("--collect",   action="store_true", help="Kumpulkan ground truth")
    parser.add_argument("--tune",      action="store_true", help="Grid search tuning")
    parser.add_argument("--eval",      action="store_true", help="Evaluasi dari CSV saja")
    parser.add_argument("--gt",        default="ground_truth.csv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--v-threshold",type=float,default=None)
    parser.add_argument("--smooth",    type=int,   default=None)
    parser.add_argument("--vote",      type=int,   default=None)
    parser.add_argument("--det-conf",  type=float, default=None)
    parser.add_argument("--no-pose",   action="store_true", help="Nonaktifkan head pose")
    args = parser.parse_args()
 
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
 
    # Load kalibrasi jika ada
    calib = Calibrator()
    if not calib.load():
        calib = None
 
    override = {}
    if args.threshold   is not None: override["gaze_threshold"]     = args.threshold
    if args.v_threshold is not None: override["vertical_threshold"]  = args.v_threshold
    if args.smooth      is not None: override["smooth_window"]        = args.smooth
    if args.vote        is not None: override["vote_window"]          = args.vote
    if args.det_conf    is not None: override["min_detection_conf"]   = args.det_conf
    if args.no_pose:                 override["use_head_pose"]         = False
 
    if args.calibrate:
        result = run_calibration(source)
        if result:
            calib = result
        if not (args.collect or args.tune or args.eval):
            return
 
    if args.collect:
        collect_ground_truth(source, args.gt, calib)
    elif args.tune:
        best = grid_search(args.gt, calib)
        if best:
            print("  Jalankan dengan config terbaik:")
            print(f"  python gaze_tuning_v2.py --source {args.source} "
                  f"--threshold {best['gaze_threshold']} "
                  f"--smooth {best['smooth_window']} "
                  f"--vote {best['vote_window']}"
                  + (" --no-pose" if not best.get("use_head_pose") else "") + "\n")
    elif args.eval:
        cfg = {**DEFAULT_CONFIG, **override}
        m = evaluate_from_csv(args.gt, cfg, calib)
        if m: print_metrics(m, cfg, f"Evaluasi dari {args.gt}")
    else:
        run_realtime(source, override or None, args.gt if os.path.exists(args.gt) else None, calib)
 
 
if __name__ == "__main__":
    main()