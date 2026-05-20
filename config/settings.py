# ─────────────────────────────────────────────
# config/settings.py
# Konfigurasi threshold & parameter sistem Eye Gaze Detection
# ─────────────────────────────────────────────

# ── Gaze Detection ──────────────────────────
GAZE_THRESHOLD      = 0.15   # Sensitivitas horizontal (range 0.05 – 0.30)
VERTICAL_THRESHOLD  = 0.15   # Sensitivitas vertikal
MAX_FACES           = 5      # Jumlah wajah maks per frame

# ── MediaPipe FaceMesh ───────────────────────
MIN_DETECTION_CONF  = 0.5    # Confidence minimum deteksi awal wajah
MIN_TRACKING_CONF   = 0.5    # Confidence minimum tracking antar frame

# ── Smoothing & Voting ───────────────────────
SMOOTH_WINDOW       = 5      # Jumlah frame untuk moving average iris ratio
VOTE_WINDOW         = 9      # Jumlah frame untuk temporal voting label

# ── Head Pose ────────────────────────────────
USE_HEAD_POSE       = True   # Aktifkan estimasi orientasi kepala
POSE_YAW_LIMIT      = 12.0   # Batas yaw (derajat) untuk dianggap "lurus"
POSE_PITCH_LIMIT    = 10.0   # Batas pitch (derajat)

# ── Kalibrasi ────────────────────────────────
USE_CALIBRATION     = True   # Aktifkan normalisasi per-user
CALIB_FILE          = "gaze_calibration.json"
CALIB_DURATION_SEC  = 3.0    # Durasi rekam baseline kalibrasi

# ── Logging ──────────────────────────────────
LOG_INTERVAL_SEC    = 60     # Interval simpan ringkasan attention rate (detik)
LOG_DIR             = "output/logs"
LOG_FILENAME        = "attention_log.csv"

# ── REST API ─────────────────────────────────
API_HOST            = "0.0.0.0"
API_PORT            = 5000
API_DEBUG           = False

# ── Video ────────────────────────────────────
DEFAULT_SOURCE      = 0      # 0=webcam, 'file.mp4', atau 'rtsp://...'
DISPLAY_WINDOW      = True   # Tampilkan jendela OpenCV
WINDOW_TITLE        = "Eye Gaze Detector"
