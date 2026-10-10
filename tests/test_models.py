import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


from emgpred.models.gvt import build_gvt  # noqa: E402
from emgpred.models.selector import build_autopruning_model, topk_mask  # noqa: E402

SMALL = dict(d_model=16, n_heads=2, n_layers_time=1, n_layers_space=1, mlp_ratio=2.0, dropout=0.0)


@pytest.mark.parametrize("pooling", ["cls", "max", "gap"])
@pytest.mark.parametrize("pos", ["learned", "sinu", "hybrid"])
def test_gvt_shapes(pooling, pos):
    m = build_gvt(20, 8, 1, 5, pooling_method=pooling, pos_embedding_method=pos, **SMALL)
    assert m(np.zeros((3, 20, 8, 1), np.float32)).shape == (3, 5)


def test_gvt_save_load_roundtrip(tmp_path):
    import keras
    m = build_gvt(20, 8, 1, 5, **SMALL)
    x = np.random.default_rng(0).normal(size=(2, 20, 8, 1)).astype(np.float32)
    m.save(tmp_path / "m.keras")
    m2 = keras.models.load_model(tmp_path / "m.keras")
    np.testing.assert_allclose(m(x), m2(x), atol=1e-6)


def test_autopruning_masks_k_channels():
    prior = np.full((8, 12), 1 / 12, np.float32)
    model, gate, sel = build_autopruning_model(prior, time_steps=20, n_ch=12, backbone=SMALL)
    x = np.random.default_rng(0).normal(size=(4, 20, 12, 1)).astype(np.float32)
    sel.k.assign(4.0)
    masked, _, _ = sel(x, training=False)
    assert np.all((np.abs(masked.numpy()).sum(axis=(1, 3)) > 0).sum(axis=1) == 4)
    assert set(model(x)) == {"angles", "task"}


def test_topk_mask():
    m = topk_mask(np.array([[0.1, 0.9, 0.5, 0.7]]), 2)
    np.testing.assert_array_equal(m, [[0, 1, 0, 1]])


@pytest.mark.parametrize("name", ["conv2plus1d", "resnet50", "pet", "raulnet"])
def test_baselines_match_paper_parameter_counts(name):
    from emgpred.models.baselines import EXPECTED_PARAMS, MODELS

    model = MODELS[name]()
    assert model.count_params() == EXPECTED_PARAMS[name]
    assert model.input_shape == (None, 100, 128, 1) and model.output_shape == (None, 5)


def test_augmentation_inactive_at_inference_and_electrode_shift():
    from emgpred.models.augmentation import AUG_NAMES, augmentation_tag, input_block
    import keras

    x = np.random.default_rng(0).normal(size=(4, 100, 128, 1)).astype(np.float32)
    sel = [3, 70, 100]
    inp = keras.Input((100, 128, 1))
    model = keras.Model(inp, input_block(inp, AUG_NAMES, selection=sel))
    np.testing.assert_allclose(model(x, training=False), x[:, :, sel])        # inferência: só seleciona
    out = model(x, training=True)
    assert out.shape == (4, 100, 3, 1) and not np.allclose(out, x[:, :, sel])
    assert augmentation_tag(["eletrodo", "amplitude"]) == "_aug-com-amplitude-eletrodo"
