from pathlib import Path
import hashlib, json, sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
ERY=ROOT.parent/'RBC_ERYTHROSIGHT_EXTERNAL_VALIDATION_20261030'
sys.dont_write_bytecode=True
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rank(s):return hashlib.sha256(s.encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(path,obj,exclusive=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def verify_freeze():
    f=read(ROOT/'audit/PRETRAIN_FREEZE.json')
    for e in f['files']:assert sha(ROOT/e['path'])==e['sha256'],f"Frozen input changed: {e['path']}"
    return f
def verify_history():
    p=read(ROOT/'audit/HISTORICAL_PRESERVATION_BASELINE.json')
    for e in p['files']:assert sha(e['path'])==e['sha256'],f"Historical file changed: {e['path']}"
    return len(p['files'])
