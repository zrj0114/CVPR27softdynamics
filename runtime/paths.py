"""Separate the checked-in physics source from local, untracked assets."""
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('SOFTDYNAMICS_WORKSPACE', CODE.parent / 'data' / 'workspace')).resolve()
DATA = Path(os.environ.get('SOFTDYNAMICS_CAPTURE_ROOT', ROOT / 'captures')).resolve()
