from pathlib import Path
import shutil

import pytest

from softdynamics.cma_benchmark import prepare
from softdynamics.io import read_json, sha256, write_json


def recovery_fixture(root, runtime):
    root.mkdir()
    checkpoint = root/'model.pt'
    checkpoint.write_bytes(b'fixture-checkpoint-identity-only')
    write_json(root/'plan.json', dict(runtime=str(runtime), config={}, objects=[],
               conditions=[dict(key='real')], checkpoint=str(checkpoint),
               model_sha256=sha256(checkpoint)))


def test_cma_prepare_creates_the_runtime_it_records(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    previous, output = tmp_path/'previous', tmp_path/'cma'
    recovery_fixture(previous, repo/'runtime')
    plan = prepare(output, previous)
    assert (Path(plan['runtime'])/'run_episode.py').is_file()
    assert (output/'code/softdynamics/cma_benchmark.py').is_file()
    assert read_json(output/'physics_parity.json')['trajectory_replay_performed'] is False
    manifest = read_json(output/'code_manifest.json')['source_files']
    for name, digest in manifest.items():
        assert sha256(output/'code'/name) == digest
    assert prepare(output, previous) == plan


def test_cma_prepare_refuses_to_relabel_changed_physics_as_same_cache(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    old_code = tmp_path/'old_code'
    shutil.copytree(repo/'runtime', old_code/'runtime', ignore=shutil.ignore_patterns('__pycache__'))
    with (old_code/'runtime/run_episode.py').open('a') as stream:
        stream.write('\n# different runtime revision\n')
    previous = tmp_path/'previous'
    recovery_fixture(previous, old_code/'runtime')
    with pytest.raises(ValueError, match='Physics differs'):
        prepare(tmp_path/'cma', previous)
    assert not (tmp_path/'cma/plan.json').exists()
