import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")

for _path in (BACKEND_DIR, SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
