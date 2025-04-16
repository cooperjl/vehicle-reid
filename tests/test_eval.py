import numpy as np

from vehicle_reid.eval import eval_veri

# Inspired by original evaluation implementation tests.
# https://github.com/Cysu/open-reid/blob/3293ca79a07ebee7f995ce647aafa7df755207b8/test/evaluation_metrics/test_cmc.py#L7


def test_eval_veri():
    """Test that the eval_veri function gives expected outputs."""
    distmat = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
        ]
    )

    q_pids = np.arange(distmat.shape[0])
    g_pids = np.arange(distmat.shape[1])
    q_camids = np.zeros(distmat.shape[0]).astype(np.int32)
    g_camids = np.ones(distmat.shape[1]).astype(np.int32)

    cmc, mAP = eval_veri(distmat, q_pids, g_pids, q_camids, g_camids)

    assert np.array_equal(
        cmc[:5], np.array([0.6, 0.6, 0.8, 1.0, 1.0], dtype=np.float32)
    )
    assert mAP == 0.7166666666666667


def test_duplicate_ids():
    """Test that the eval_veri function gives expected outputs with duplicate ids."""
    distmat = np.tile(np.arange(4), (4, 1))
    q_pids = np.array([0, 0, 1, 1])
    g_pids = np.array([0, 0, 1, 1])
    q_camids = np.zeros(distmat.shape[0]).astype(np.int32)
    g_camids = np.ones(distmat.shape[1]).astype(np.int32)

    cmc, _ = eval_veri(distmat, q_pids, g_pids, q_camids, g_camids)
    assert np.array_equal(cmc[:4], np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32))
