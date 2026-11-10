"""Keep process-local physics adapters isolated between workflow tests."""

import pytest


@pytest.fixture(autouse=True)
def restore_physics_adapter_globals():
    # The CLI runs elastic and legacy workflows in separate processes. Tests
    # exercise both in one interpreter, so restore the adapter on every exit.
    from softdynamics import joint_acceleration, recovery_data

    original_key = recovery_data.qkey
    original_folder = joint_acceleration.trajectory_folder
    try:
        yield
    finally:
        recovery_data.qkey = original_key
        joint_acceleration.trajectory_folder = original_folder
