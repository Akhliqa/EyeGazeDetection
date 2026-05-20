"""
utils/logger.py
Logging attention rate ke CSV dan/atau database SQLite
"""

import csv
import os
import time
import sqlite3
from datetime import datetime
from config import settings


class AttentionLogger:
    """
    Mencatat data attention rate per interval waktu ke file CSV.
    Opsional: juga simpan ke database SQLite untuk query lanjutan.
    """

    def __init__(self, log_dir: str = settings.LOG_DIR,
                 filename: str = settings.LOG_FILENAME,
                 interval_sec: float = settings.LOG_INTERVAL_SEC,
                 use_db: bool = False):

        self.interval_sec = interval_sec
        self.use_db = use_db
        self._last_log_time = time.time()

        # Pastikan direktori ada
        os.makedirs(log_dir, exist_ok=True)

        # CSV
        self.csv_path = os.path.join(log_dir, filename)
        self._init_csv()

        # SQLite (opsional)
        if use_db:
            db_name = filename.replace(".csv", ".db")
            self.db_path = os.path.join(log_dir, db_name)
            self._init_db()

        print(f"[Logger] CSV  : {self.csv_path}")
        if use_db:
            print(f"[Logger] DB   : {self.db_path}")
        print(f"[Logger] Interval: {interval_sec}s")

    # ── Inisialisasi ────────────────────────────────────────────────────────

    def _init_csv(self):
        write_header = not os.path.exists(self.csv_path)
        self._csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=[
            "timestamp", "frame_number", "total_faces",
            "looking_count", "attention_rate"
        ])
        if write_header:
            self._writer.writeheader()
            self._csv_file.flush()

    def _init_db(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS attention_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT,
                frame_number   INTEGER,
                total_faces    INTEGER,
                looking_count  INTEGER,
                attention_rate REAL
            )
        """)
        self._conn.commit()

    # ── Logging ─────────────────────────────────────────────────────────────

    def log(self, frame_number: int, total_faces: int,
            looking_count: int, attention_rate: float,
            force: bool = False):
        """
        Tulis satu baris log.
        Jika force=False, hanya tulis setiap interval_sec detik.
        """
        now = time.time()
        if not force and (now - self._last_log_time) < self.interval_sec:
            return

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp":      ts,
            "frame_number":   frame_number,
            "total_faces":    total_faces,
            "looking_count":  looking_count,
            "attention_rate": round(attention_rate, 2),
        }

        # CSV
        self._writer.writerow(row)
        self._csv_file.flush()

        # SQLite
        if self.use_db:
            self._conn.execute(
                "INSERT INTO attention_log "
                "(timestamp, frame_number, total_faces, looking_count, attention_rate) "
                "VALUES (?,?,?,?,?)",
                (ts, frame_number, total_faces, looking_count, round(attention_rate, 2))
            )
            self._conn.commit()

        self._last_log_time = now

    # ── Query (SQLite) ───────────────────────────────────────────────────────

    def query_daily(self, date_str: str = None):
        """
        Ambil semua log untuk tanggal tertentu (format: YYYY-MM-DD).
        Memerlukan use_db=True.
        """
        if not self.use_db:
            raise RuntimeError("Database tidak aktif. Buat logger dengan use_db=True.")
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        cur = self._conn.execute(
            "SELECT * FROM attention_log WHERE timestamp LIKE ?",
            (f"{date_str}%",)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def average_attention(self, date_str: str = None):
        """Rata-rata attention rate untuk satu hari (memerlukan use_db=True)."""
        rows = self.query_daily(date_str)
        if not rows:
            return 0.0
        return sum(r["attention_rate"] for r in rows) / len(rows)

    # ── Penutupan ────────────────────────────────────────────────────────────

    def close(self):
        self._csv_file.close()
        if self.use_db and hasattr(self, "_conn"):
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
