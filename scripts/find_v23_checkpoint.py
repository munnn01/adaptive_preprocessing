"""Resolve the immutable completed V23 AR checkpoint in a Kaggle input mount."""
import hashlib
from pathlib import Path
import sys

V23_AR_SHA256 = '9c1dae49253b1d54d0caec0032e5075e59953d882e83b013c1534a985d1e0839'


def resolve(root):
    matches = [p for p in Path(root).rglob('preprocessor_last.pth')
               if hashlib.sha256(p.read_bytes()).hexdigest() == V23_AR_SHA256]
    if not matches:
        raise FileNotFoundError('pinned V23 AR checkpoint is missing; refusing an unmatched baseline')
    return sorted(matches)[0]


if __name__ == '__main__':
    print(resolve(sys.argv[1]))
