"""
main.py — Entry point utama Eye Gaze Detection System
======================================================
Cara pakai:
  python main.py                              # webcam, tanpa API
  python main.py --source rekaman.mp4         # dari file video
  python main.py --source rtsp://192.168.1.100:554/stream   # IP Camera
  python main.py --source 0 --api             # aktifkan REST API
  python main.py --source 0 --db              # aktifkan logging ke SQLite
  python main.py --source 0 --no-display      # tanpa jendela OpenCV (headless)
"""

import argparse
import time
import cv2
import numpy as np
from collections import deque
from datetime import datetime

from gaze_detector import run as run_simple          # versi dasar
from config import settings
from utils.video_source import VideoSource
from utils.visualizer import draw_hud, draw_face, draw_attention_bar
from utils.logger import AttentionLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eye Gaze Detection System",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Contoh:
  python main.py --source 0
  python main.py --source video.mp4 --no-display
  python main.py --source 0 --api --db
  python main.py --source rtsp://192.168.1.100:554/stream --api
        """
    )
    parser.add_argument("--source",     default=str(settings.DEFAULT_SOURCE),
                        help="Sumber video: 0=webcam, path file, atau RTSP URL")
    parser.add_argument("--api",        action="store_true",
                        help="Aktifkan REST API Flask")
    parser.add_argument("--db",         action="store_true",
                        help="Simpan log ke SQLite (selain CSV)")
    parser.add_argument("--no-display", action="store_true",
                        help="Jalankan tanpa jendela OpenCV (headless/server)")
    parser.add_argument("--threshold",  type=float, default=None,
                        help=f"Override gaze_threshold (default: {settings.GAZE_THRESHOLD})")
    parser.add_argument("--smooth",     type=int,   default=None,
                        help=f"Override smooth_window (default: {settings.SMOOTH_WINDOW})")
    parser.add_argument("--no-pose",    action="store_true",
                        help="Nonaktifkan head pose estimation")
    parser.add_argument("--simple",     action="store_true",
                        help="Gunakan versi sederhana (gaze_detector.py)")
    return parser.parse_args()


def resolve_source(source_str: str):
    try:
        return int(source_str)
    except ValueError:
        return source_str


def main():
    args = parse_args()
    source = resolve_source(args.source)

    # ── Mode sederhana ──────────────────────────────────────────────────────
    if args.simple:
        print("[Main] Menjalankan mode sederhana (gaze_detector.py)...")
        run_simple(source=source)
        return

    # ── Konfigurasi ─────────────────────────────────────────────────────────
    config = {
        "gaze_threshold":     args.threshold if args.threshold else settings.GAZE_THRESHOLD,
        "vertical_threshold": settings.VERTICAL_THRESHOLD,
        "smooth_window":      args.smooth if args.smooth else settings.SMOOTH_WINDOW,
        "vote_window":        settings.VOTE_WINDOW,
        "use_head_pose":      not args.no_pose and settings.USE_HEAD_POSE,
        "use_calibration":    settings.USE_CALIBRATION,
        "max_faces":          settings.MAX_FACES,
        "min_detection_conf": settings.MIN_DETECTION_CONF,
        "min_tracking_conf":  settings.MIN_TRACKING_CONF,
        "pose_yaw_limit":     settings.POSE_YAW_LIMIT,
        "pose_pitch_limit":   settings.POSE_PITCH_LIMIT,
    }

    # ── Import detektor versi lengkap ───────────────────────────────────────
    from gaze_tuning_v2 import ImprovedGazeDetector, Calibrator

    # Load kalibrasi jika ada
    calib = Calibrator()
    if not calib.load(settings.CALIB_FILE):
        print("[Main] Kalibrasi belum ada. Jalankan gaze_tuning_v2.py --calibrate "
              "untuk hasil terbaik.")
        calib = None

    detector = ImprovedGazeDetector(config, calib)

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = AttentionLogger(
        log_dir=settings.LOG_DIR,
        filename=settings.LOG_FILENAME,
        interval_sec=settings.LOG_INTERVAL_SEC,
        use_db=args.db,
    )

    # ── REST API ─────────────────────────────────────────────────────────────
    if args.api:
        from api.server import run_server, update_state, set_logger
        set_logger(logger)
        run_server()
        update_state(
            running=True,
            source=source,
            started_at=datetime.now().isoformat(),
            config=config,
        )
    else:
        update_state = lambda **kw: None   # no-op jika API tidak aktif

    # ── Loop Utama ───────────────────────────────────────────────────────────
    print(f"\n[Main] Memulai deteksi dari: {source}")
    print(f"[Main] Config: {config}")
    print("[Main] Tekan Q untuk keluar.\n")

    fps_buf       = deque(maxlen=30)
    attn_total    = 0
    frames_total  = 0
    frame_number  = 0

    try:
        with VideoSource(source) as vs:
            for frame in vs:
                frame_number  += 1
                frames_total  += 1

                t0    = time.perf_counter()
                faces = detector.process_frame(frame)
                ms    = (time.perf_counter() - t0) * 1000
                fps_buf.append(1000 / (ms + 1e-9))

                looking_now  = sum(f["looking"] for f in faces)
                attn_total  += 1 if looking_now > 0 else 0
                attn_rate    = attn_total / frames_total * 100
                fps_now      = float(np.mean(fps_buf))

                # Annotasi
                if not args.no_display or args.api:
                    for f in faces:
                        draw_face(frame, f)
                    draw_hud(frame, attn_rate, looking_now,
                             len(faces), fps_now, ms, config)
                    draw_attention_bar(frame, attn_rate)

                # Log
                logger.log(frame_number, len(faces), looking_now, attn_rate)

                # Update state API
                update_state(
                    frame_number=frame_number,
                    total_faces=len(faces),
                    looking_count=looking_now,
                    attention_rate=attn_rate,
                    fps=round(fps_now, 1),
                    latency_ms=round(ms, 1),
                )

                # Tampilkan
                if not args.no_display:
                    cv2.imshow(settings.WINDOW_TITLE, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except RuntimeError as e:
        print(f"[Main] Error: {e}")
    finally:
        detector.close()
        logger.close()
        if not args.no_display:
            cv2.destroyAllWindows()

        print(f"\n[Main] Selesai.")
        print(f"  Total frame   : {frames_total}")
        print(f"  Attention rate: {attn_rate:.1f}%")
        print(f"  Rata-rata FPS : {float(np.mean(fps_buf)):.1f}")
        print(f"  Log CSV       : {logger.csv_path}")


if __name__ == "__main__":
    main()
