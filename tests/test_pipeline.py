import numpy as np

from garch_groundroll import (kmeans, load_config, load_dataset, make_synthetic_shot,
                              run_pipeline, select_signal_clusters, true_snr_db)
from garch_groundroll.features import build_features, gabor_kernel


def test_synthetic_geometry():
    g = make_synthetic_shot()
    assert g.data.shape == (1501, 100)
    i = np.argmax(g.signal[:350, 50])  # first reflection, offset +20 m
    assert abs(i * g.dt - np.hypot(0.5, 20 / 1800)) < 0.004
    np.testing.assert_allclose(g.data, g.signal + g.ground_roll)


def test_gabor_kernel_shape_and_orientation():
    k0 = gabor_kernel(8, 0)
    assert k0.shape[0] == k0.shape[1] and k0.shape[0] % 2 == 1
    # 0 deg: the carrier varies along columns only at the centre row
    r = k0.shape[0] // 2
    assert np.allclose(np.angle(k0[:, r]), 0)


def test_features_standardised():
    csd = np.random.default_rng(0).random((60, 40))
    X = build_features(csd, [3, 6], [0, 90], 1.5, True, "clip")
    assert X.shape == (2400, 1 + 4 + 2)
    np.testing.assert_allclose(X.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(X.std(0), 1, atol=1e-10)


def test_kmeans_labels_by_size():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.standard_normal((300, 2)), rng.standard_normal((100, 2)) + 8])
    lab = kmeans(X, 2)
    assert np.all(lab[:300] == 1) and np.all(lab[300:] == 2)


def test_cluster_selection_rules():
    labels = np.array([[1, 1, 2, 2], [3, 3, 4, 4]])
    csd = np.array([[0.1, 0.1, 1.0, 1.0], [0.2, 0.2, 0.9, 0.9]])
    mask, kept = select_signal_clusters(labels, csd, "gap")
    assert kept.tolist() == [1, 3]
    np.testing.assert_array_equal(mask, [[1, 1, 0, 0], [1, 1, 0, 0]])
    assert select_signal_clusters(labels, csd, "n_keep", n_keep=1)[1].tolist() == [1]
    assert select_signal_clusters(labels, csd, "manual", clusters=[2])[1].tolist() == [2]
    assert select_signal_clusters(labels, csd, "threshold", threshold=0.5)[1].tolist() == [1, 3]


def test_pipeline_improves_true_snr_on_synthetic():
    g = make_synthetic_shot()
    cfg = load_config("synthetic")
    cfg.pop("dataset")
    r = run_pipeline(g.data, g.dt, g.dx, cfg)
    before, after = true_snr_db(r.filtered, g.signal), true_snr_db(r.corrected, g.signal)
    assert after > before + 1.0
    np.testing.assert_allclose(r.corrected, r.filtered + r.recovered, atol=1e-10)
    np.testing.assert_allclose(r.data, r.corrected + r.corrected_noise, atol=1e-10)


def test_bundled_datasets_load():
    for name in ["shot_gather_502", "field_shot", "stacked_section", "crossline_3d"]:
        ds = load_dataset(name)
        assert ds.data.ndim == 2 and ds.dt > 0 and ds.dx > 0
        if ds.fk_mask is not None:
            assert ds.fk_mask.shape == ds.data.shape
