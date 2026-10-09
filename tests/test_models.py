"""Covers synopsis TC4: all six models train without error and produce
valid class predictions/probabilities. Fits each model directly (no GA
search) to keep this fast; GA-tuned end-to-end training is covered by the
slower smoke test in test_run_pipeline.py.
"""
from pipeline.models import build_model_registry

EXPECTED_MODEL_NAMES = {"xgb", "gradient_boosting", "random_forest", "lightgbm", "catboost", "ebm"}


def test_registry_has_all_six_synopsis_models():
    registry = build_model_registry()
    assert set(registry.keys()) == EXPECTED_MODEL_NAMES


def test_every_model_fits_and_predicts_probabilities(small_split):
    (x_train, x_test, y_train, y_test), _ = small_split
    registry = build_model_registry()

    for name, (model, _param_grid) in registry.items():
        model.fit(x_train, y_train)
        proba = model.predict_proba(x_test)
        assert proba.shape == (len(x_test), 2), f"{name} produced unexpected proba shape"
        assert (proba >= 0).all() and (proba <= 1).all(), f"{name} produced out-of-range probabilities"
