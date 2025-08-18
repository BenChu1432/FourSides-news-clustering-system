import json

def _json_str(val):
    if val is None:
        return ""
    try:
        return json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)