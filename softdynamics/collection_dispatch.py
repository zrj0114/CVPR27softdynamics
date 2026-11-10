"""Route new collectors to an immutable code revision; keep running jobs intact."""
import json
import os
import sys
from pathlib import Path


def execution_target(root, current_code):
    settings=Path(root)/'acceleration.json'
    if not settings.exists():return None
    requested=json.loads(settings.read_text()).get('execution_code_root')
    if requested is None:return None
    target=Path(requested).resolve()
    if target==Path(current_code).resolve():return None
    if not (target/'softdynamics/elastic_data.py').is_file():
        raise ValueError('Missing collection execution revision: '+str(target))
    return target


def redirect_collector(root):
    target=execution_target(root,Path(__file__).resolve().parents[1])
    if target is None:return
    env=dict(os.environ)
    env['PYTHONPATH']=str(target)+os.pathsep+env.get('PYTHONPATH','')
    os.chdir(target)
    os.execve(sys.executable,[sys.executable,'-u','-m','softdynamics.elastic_data',*sys.argv[1:]],env)
