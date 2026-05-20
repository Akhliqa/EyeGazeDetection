"""
api/server.py
REST API berbasis Flask untuk integrasi dashboard eksternal.
Jalankan bersama main.py dengan flag --api, atau standalone untuk testing.

Endpoint:
  GET  /status           -> Status sistem & jumlah kamera aktif
  GET  /metrics          -> Attention rate real-time semua kamera
  GET  /report?date=     -> Laporan harian format JSON
  POST /config           -> Update threshold & parameter secara remote
"""

import threading
from datetime import datetime
from flask import Flask, jsonify, request
from config import settings

app = Flask(__name__)

# ── Shared State (diakses dari thread utama dan API) ─────────────────────────
_state = {
    "running":        False,
    "source":         None,
    "frame_number":   0,
    "total_faces":    0,
    "looking_count":  0,
    "attention_rate": 0.0,
    "fps":            0.0,
    "latency_ms":     0.0,
    "config": {
        "gaze_threshold":     settings.GAZE_THRESHOLD,
        "vertical_threshold": settings.VERTICAL_THRESHOLD,
        "smooth_window":      settings.SMOOTH_WINDOW,
        "vote_window":        settings.VOTE_WINDOW,
        "use_head_pose":      settings.USE_HEAD_POSE,
        "max_faces":          settings.MAX_FACES,
    },
    "started_at": None,
}
_state_lock = threading.Lock()
_logger = None   # Diisi oleh main.py jika diaktifkan


def update_state(**kwargs):
    """Dipanggil dari thread deteksi untuk memperbarui state real-time."""
    with _state_lock:
        _state.update(kwargs)


def set_logger(logger_instance):
    global _logger
    _logger = logger_instance


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    """Status sistem dan informasi kamera aktif."""
    with _state_lock:
        return jsonify({
            "status":     "running" if _state["running"] else "stopped",
            "source":     str(_state["source"]),
            "started_at": _state["started_at"],
            "uptime_sec": (
                (datetime.now() - datetime.fromisoformat(_state["started_at"])).seconds
                if _state["started_at"] else 0
            ),
            "fps":        _state["fps"],
            "latency_ms": _state["latency_ms"],
        })


@app.get("/metrics")
def metrics():
    """Attention rate real-time dari kamera yang aktif."""
    with _state_lock:
        return jsonify({
            "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_number":   _state["frame_number"],
            "total_faces":    _state["total_faces"],
            "looking_count":  _state["looking_count"],
            "attention_rate": round(_state["attention_rate"], 2),
            "fps":            _state["fps"],
        })


@app.get("/report")
def report():
    """
    Laporan harian dalam format JSON.
    Query param: ?date=YYYY-MM-DD (default: hari ini)
    """
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    if _logger is None or not _logger.use_db:
        return jsonify({
            "error": "Logger database tidak aktif. Jalankan dengan --db flag."
        }), 503

    try:
        rows = _logger.query_daily(date_str)
        avg  = _logger.average_attention(date_str)
        return jsonify({
            "date":             date_str,
            "total_records":    len(rows),
            "avg_attention":    round(avg, 2),
            "data":             rows,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/config")
def update_config():
    """
    Update parameter deteksi secara remote.
    Body JSON contoh:
      { "gaze_threshold": 0.13, "smooth_window": 7 }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Body JSON tidak valid."}), 400

    allowed = {
        "gaze_threshold", "vertical_threshold", "smooth_window",
        "vote_window", "use_head_pose", "max_faces",
        "min_detection_conf", "min_tracking_conf",
    }
    updated = {}
    with _state_lock:
        for k, v in body.items():
            if k in allowed:
                _state["config"][k] = v
                updated[k] = v

    if not updated:
        return jsonify({"error": f"Tidak ada parameter valid. Gunakan: {allowed}"}), 400

    return jsonify({"updated": updated, "config": _state["config"]})


# ── Runner ───────────────────────────────────────────────────────────────────

def run_server(host: str = settings.API_HOST,
               port: int = settings.API_PORT,
               debug: bool = settings.API_DEBUG):
    """
    Jalankan Flask server di thread terpisah agar tidak memblokir loop utama.
    """
    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=debug, use_reloader=False),
        daemon=True,
        name="flask-api"
    )
    thread.start()
    print(f"[API] Server berjalan di http://{host}:{port}")
    return thread


# ── Standalone testing ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[API] Menjalankan server standalone (mode testing)...")
    _state["running"] = True
    _state["started_at"] = datetime.now().isoformat()
    app.run(host=settings.API_HOST, port=settings.API_PORT, debug=True)
