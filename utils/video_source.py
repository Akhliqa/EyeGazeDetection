"""
utils/video_source.py
Handler webcam / file video / IP stream (RTSP)
"""

import cv2


class VideoSource:
    """
    Wrapper OpenCV VideoCapture yang mendukung:
      - Webcam lokal   : source=0 (atau integer lain)
      - File video     : source='rekaman.mp4'
      - IP Camera RTSP : source='rtsp://192.168.1.100:554/stream'
    """

    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[VideoSource] Tidak bisa membuka source: {self.source}")
        print(f"[VideoSource] Terbuka: {self.source}  "
              f"({self.width}x{self.height} @ {self.fps:.1f} FPS)")

    # ── Properties ──────────────────────────

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    @property
    def is_file(self):
        return isinstance(self.source, str) and not self.source.startswith("rtsp")

    @property
    def total_frames(self):
        """Hanya valid untuk sumber file video."""
        if self.is_file:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    # ── Baca Frame ──────────────────────────

    def read(self):
        """Return (success: bool, frame: ndarray | None)."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def __iter__(self):
        """Iterasi frame: for frame in source."""
        while True:
            ok, frame = self.read()
            if not ok:
                break
            yield frame

    # ── Kontrol ─────────────────────────────

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def __repr__(self):
        return (f"VideoSource(source={self.source!r}, "
                f"{self.width}x{self.height}, {self.fps:.1f}fps)")
