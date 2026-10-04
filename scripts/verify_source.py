"""Verify upload source only; compare external credential values without printing them."""
from pathlib import Path
import ast
import hashlib
import json
import sys

root = Path(sys.argv[1]).resolve()
pool = Path(sys.argv[2])
values = json.loads(pool.read_text(encoding="utf-8-sig"))
secrets = []
for value in values.values():
    if isinstance(value, str):
        secrets.append(value.encode())
    elif isinstance(value, dict):
        secrets.extend(str(v).encode() for k, v in value.items() if k in ("key", "token"))
paths = []
for directory in ("adaptive_vcm", "tests", "configs", "scripts", "docs"):
    paths.extend(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
paths.extend(root / name for name in ("README.md", "pyproject.toml", ".gitignore", ".gitattributes"))
hashes = {}
for path in sorted(paths):
    if path.suffix.lower() in (".pyc", ".pth", ".pt", ".pdf", ".mp4", ".jsonl"):
        raise RuntimeError("unexpected non-source member")
    data = path.read_bytes()
    if any(secret and secret in data for secret in secrets):
        raise RuntimeError("credential value found in upload source")
    if path.suffix == ".py":
        ast.parse(data.decode("utf-8-sig"), filename=str(path))
    hashes[path.relative_to(root).as_posix()] = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
print(json.dumps({"files": len(hashes), "bytes": sum(p.stat().st_size for p in paths),
                  "no_pool_credentials": True, "python_ast_valid": True,
                  "source_tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}, indent=2))

