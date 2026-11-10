from pathlib import Path

import pytest

from softdynamics.io import sha256, write_json
from softdynamics.parameters import ParameterSpace
from softdynamics.recovery_data import trajectory_folder
from softdynamics.shared_world_cma_benchmark import import_incumbent


def fixture(tmp_path):
    previous, root = tmp_path/'previous', tmp_path/'new'
    workspace, runtime = tmp_path/'workspace', root/'code/runtime'
    paths = dict(initial=workspace/'initial_states/episode_0.npz',
        actions=workspace/'queries/episode_0/actions.npz', asset=workspace/'asset/asset.json',
        pad_kernel=runtime/'gs-layer/pad_contact_kernels.py',
        adapter=runtime.parent/'softdynamics/adapters/pad_empm.py',
        initializer=runtime/'run_episode.py', transfer=runtime/'gs-layer/deterministic_transfer.py')
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(key)
    row = tmp_path/'prepared/complete.json'; write_json(row, dict(workspace=str(workspace)))
    data_root = tmp_path/'data'
    write_json(data_root/'plan.json', dict(source_sha256={str(row): sha256(row)}))
    config = dict(baseline=dict(E_pa=3000., nu=.3, ground_friction=.5, velocity_decay_rate_s_inv=.4),
                  coordinate_scales=[3000., .4, .5, 1.])
    entry = dict(episode=0, workspace=str(workspace), row=str(row), dtmax=.0001, domain_margin=.075,
        initial_sha256=sha256(paths['initial']), actions_sha256=sha256(paths['actions']), asset_sha256=sha256(paths['asset']),
        observation_sha256='frozen', surface_ids_sha256='frozen', real_surface_sha256='frozen')
    plan = dict(objects=[dict(object='fixture', support=[0], query=[], episodes=[entry])],
        config=config, runtime=str(runtime), data_root=str(data_root), incumbent_q=[0.]*4,
        data_plan_sha256=sha256(data_root/'plan.json'))
    write_json(previous/'plan.json', plan)
    folder = trajectory_folder(previous, 'fixture', 0, plan['incumbent_q'])
    folder.mkdir(parents=True)
    for name in ('tracks', 'surface'):
        (folder/(name+'.npy')).write_bytes(b'fixed-payload')
    write_json(folder/'validation.json', [])
    write_json(folder/'complete.json', dict(valid=True, smoke=False, q=plan['incumbent_q'],
        parameters=ParameterSpace.from_config(config).parameters(plan['incumbent_q']), dtmax=.0001,
        domain=dict(margin_m=.075), source_hashes={key: sha256(path) for key, path in paths.items()},
        payload_sha256={name: sha256(folder/(name+'.npy')) for name in ('tracks', 'surface')}, elapsed_s=1.))
    return root, plan, previous, paths, row


def test_cache_accepts_real_entry_schema_and_preserves_exact_payload(tmp_path):
    root, plan, previous, _, _ = fixture(tmp_path)
    records = import_incumbent(root, plan, previous)
    assert records[0]['physical_seconds'] == 1.
    for name in ('surface.npy', 'tracks.npy'):
        new = trajectory_folder(root, 'fixture', 0, [0.]*4)/name
        old = trajectory_folder(previous, 'fixture', 0, [0.]*4)/name
        assert new.stat().st_ino == old.stat().st_ino


@pytest.mark.parametrize('changed', ['row', 'adapter', 'surface'])
def test_cache_rejects_changed_row_solver_or_payload(tmp_path, changed):
    root, plan, previous, paths, row = fixture(tmp_path)
    target = row if changed == 'row' else paths['adapter'] if changed == 'adapter' else (
        trajectory_folder(previous, 'fixture', 0, [0.]*4)/'surface.npy')
    target.write_text('changed')
    with pytest.raises(ValueError, match='Changed|differ'):
        import_incumbent(root, plan, previous)
    assert not (root/'trajectories').exists()
