"""
utils/visualizer.py
Overlay & annotasi frame untuk Eye Gaze Detection
"""

import cv2
import numpy as np
from collections import deque


# Warna per label
LABEL_COLORS = {
    "CAMERA": (0, 230, 100),   # hijau
    "LEFT":   (0, 165, 255),   # oranye
    "RIGHT":  (0, 165, 255),   # oranye
    "UP":     (255, 200, 0),   # kuning
    "DOWN":   (255, 200, 0),   # kuning
}
COLOR_DEFAULT = (100, 100, 255)


def _color(label: str):
    for key in LABEL_COLORS:
        if key in label.upper():
            return LABEL_COLORS[key]
    return COLOR_DEFAULT


# ── HUD (Heads-Up Display) ──────────────────────────────────────────────────

def draw_hud(frame, attention_rate: float, looking_now: int,
             total_faces: int, fps: float, latency_ms: float,
             config: dict | None = None):
    """
    Gambar panel HUD semi-transparan di bagian atas frame.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (8, 8, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame,
                f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f} ms",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 220, 255), 1)

    cv2.putText(frame,
                f"Attention rate: {attention_rate:.1f}%  |  "
                f"Wajah: {total_faces}  |  Menatap: {looking_now}",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 255, 200), 1)

    if config:
        cfg_text = (f"thr={config.get('gaze_threshold', '?')}  "
                    f"smooth={config.get('smooth_window', '?')}  "
                    f"vote={config.get('vote_window', '?')}  "
                    f"pose={'on' if config.get('use_head_pose') else 'off'}")
        cv2.putText(frame, cfg_text,
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 180), 1)


# ── Annotasi Per Wajah ──────────────────────────────────────────────────────

def draw_face(frame, face_result: dict):
    """
    Gambar titik iris, label gaze, dan confidence untuk satu wajah.
    face_result: dict dari GazeDetector.process_frame()
    """
    iris_l = face_result.get("iris_l")
    iris_r = face_result.get("iris_r")
    label  = face_result.get("label", "")
    conf   = face_result.get("confidence", 0.0)
    looking = face_result.get("looking", 0)

    color = _color(label)

    # Titik iris
    if iris_l:
        cv2.circle(frame, iris_l, 4, color, -1)
    if iris_r:
        cv2.circle(frame, iris_r, 4, color, -1)

    # Label + confidence
    if iris_l:
        text = f"{label} ({conf * 100:.0f}%)"
        cv2.putText(frame, text,
                    (iris_l[0] - 50, iris_l[1] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

    # Sinyal head pose (opsional)
    signals = face_result.get("signals", {})
    if signals.get("yaw") is not None:
        pose_text = f"yaw:{signals['yaw']}  pitch:{signals['pitch']}"
        h = frame.shape[0]
        cv2.putText(frame, pose_text,
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


# ── Attention Bar ───────────────────────────────────────────────────────────

def draw_attention_bar(frame, attention_rate: float, bar_x=10, bar_y=None,
                       bar_w=200, bar_h=12):
    """
    Gambar progress bar attention rate di sudut kanan bawah frame.
    """
    h, w = frame.shape[:2]
    if bar_y is None:
        bar_y = h - 30

    filled = int((attention_rate / 100) * bar_w)
    color_bar = (0, 200, 80) if attention_rate >= 50 else (0, 120, 255)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h), color_bar, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (120, 120, 120), 1)
    cv2.putText(frame, f"{attention_rate:.1f}%",
                (bar_x + bar_w + 6, bar_y + bar_h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


# ── Label Manual (Ground Truth Collection) ─────────────────────────────────

def draw_label_status(frame, current_label: str | None, frame_count: int,
                      data_count: int):
    """
    Tampilkan label aktif saat mode pengumpulan ground truth.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = (0, 255, 128) if current_label == "CAMERA" else (0, 200, 255)
    text  = f"Label: {current_label}" if current_label else "Tekan C/L/R/U/D untuk memberi label"
    cv2.putText(frame, text, (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"Frame: {frame_count}  |  Data: {data_count}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
