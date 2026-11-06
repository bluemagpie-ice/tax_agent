"""Run pytest in a unique workspace temp directory.

Windows sandbox SIDs can differ from Python's user SID. Python's mode=0700
then creates a temp directory the sandbox cannot traverse. Only pytest's
workspace temp directories use normal inherited permissions here.
"""
import os
from pathlib import Path
import sys
import uuid

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
temp = root / '.test_runs' / ('pytest-' + uuid.uuid4().hex)
temp.parent.mkdir(exist_ok=True)
original_mkdir = os.mkdir


def mkdir(path, mode=0o777, **kwargs):
    resolved = Path(path).resolve()
    if os.name == 'nt' and mode == 0o700 and resolved.is_relative_to(temp):
        mode = 0o777
    return original_mkdir(path, mode, **kwargs)


if __name__ == '__main__':
    os.mkdir = mkdir
    import pytest
    args = sys.argv[1:] or ['tests/', '-q']
    raise SystemExit(pytest.main([*args, '--basetemp=' + str(temp)]))
