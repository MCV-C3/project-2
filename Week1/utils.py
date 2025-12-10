import hashlib
import json

def compute_hash(obj) -> str:
    """Compute a hash for any serializable Python object."""
    json_str = json.dumps(obj, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()