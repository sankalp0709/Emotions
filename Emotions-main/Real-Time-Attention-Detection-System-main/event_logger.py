import json
import time
import datetime

def _mmss_from_ms(ms: int) -> str:
    if ms is None or ms < 0:
        return None
    s = int(ms // 1000)
    m = s // 60
    s = s % 60
    return f"{m:02d}:{s:02d}"

def _mmss_from_secs(secs: float) -> str:
    if secs is None or secs < 0:
        return None
    s = int(secs)
    m = s // 60
    s = s % 60
    return f"{m:02d}:{s:02d}"

def log_event(emotion: str, action: str, player, session_start: float, confidence: float, path: str = 'events_log.jsonl'):
    try:
        iso = datetime.datetime.utcnow().isoformat() + "Z"
        entry = {
            "timestamp": iso,
            "emotion": emotion,
            "confidence": round(float(confidence), 2),
            "action": action
        }
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
