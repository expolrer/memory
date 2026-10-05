import numpy as np

from data_generation.export_dp3_zarr import export_dp3_zarr
from data_generation.validate_dp3_zarr import validate_zarr


def test_export_dp3_zarr_round_trip(tmp_path):
    npz_path = tmp_path / "demo.npz"
    out_path = tmp_path / "demo.zarr"
    joint_states = np.zeros((2, 5, 13), dtype=np.float32)
    for episode in range(2):
        for t in range(5):
            joint_states[episode, t, 0] = t / 4.0
            joint_states[episode, t, 1] = episode
            joint_states[episode, t, 10] = 0.05 * episode
    actions = np.zeros_like(joint_states)
    actions[:, :-1] = joint_states[:, 1:] - joint_states[:, :-1]
    actions[:, -1] = actions[:, -2]
    np.savez_compressed(
        npz_path,
        joint_states=joint_states,
        actions=actions,
        rule_ids=np.asarray(["rule001", "rule020"]),
        success=np.asarray([True, True]),
        process_scores=np.asarray([1.0, 1.0], dtype=np.float32),
        joint_source="primitive",
        rule_profile="paper_fidelity",
    )

    summary = export_dp3_zarr(npz_path, out_path, point_count=16, overwrite=True)
    assert summary["state_shape"] == [10, 13]
    assert summary["action_shape"] == [10, 13]
    assert summary["point_cloud_shape"] == [10, 16, 6]
    assert summary["rule_profile"] == "paper_fidelity"

    validation = validate_zarr(out_path)
    assert validation["episodes"] == 2
    assert validation["total_steps"] == 10
    assert validation["point_cloud_shape"] == [10, 16, 6]
