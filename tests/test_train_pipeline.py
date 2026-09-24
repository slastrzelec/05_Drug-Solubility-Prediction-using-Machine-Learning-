"""Tests for the leakage guardrail described in SPEC.md: scaling and
feature selection must be fit on the training fold only, never on the
full dataset before splitting.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from train_v2 import make_pipeline


def _synthetic_data(n_samples=60, n_features=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=5.0, scale=2.0, size=(n_samples, n_features))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(scale=0.1, size=n_samples)
    return X, y


def test_pipeline_has_scaler_selector_model_in_order():
    pipeline = make_pipeline(RandomForestRegressor(random_state=0))
    assert list(pipeline.named_steps.keys()) == ["scaler", "selector", "model"]


def test_scaler_is_fit_on_training_fold_only_not_full_dataset():
    X, y = _synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipeline = make_pipeline(RandomForestRegressor(n_estimators=10, random_state=0))
    pipeline.fit(X_train, y_train)

    fitted_scaler_mean = pipeline.named_steps["scaler"].mean_
    train_only_mean = X_train.mean(axis=0)
    full_dataset_mean = X.mean(axis=0)

    # The fitted scaler must match the train-only statistics...
    assert np.allclose(fitted_scaler_mean, train_only_mean)
    # ...and must NOT match statistics computed over train+test, which
    # would mean the test set leaked into preprocessing.
    assert not np.allclose(fitted_scaler_mean, full_dataset_mean)


def test_predict_on_test_set_does_not_refit_scaler_or_selector():
    X, y = _synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipeline = make_pipeline(RandomForestRegressor(n_estimators=10, random_state=0))
    pipeline.fit(X_train, y_train)

    mean_before = pipeline.named_steps["scaler"].mean_.copy()
    support_before = pipeline.named_steps["selector"].get_support().copy()

    pipeline.predict(X_test)

    assert np.array_equal(pipeline.named_steps["scaler"].mean_, mean_before)
    assert np.array_equal(pipeline.named_steps["selector"].get_support(), support_before)
