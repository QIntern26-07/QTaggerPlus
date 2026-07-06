import optuna
import pytest
from classical import models


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_make_model_returns_fittable_estimator(name):
    est = models.make_model(name, params={}, task="binary")
    assert hasattr(est, "fit") and hasattr(est, "predict")


def test_svm_has_probability_enabled():
    est = models.make_model("svm", params={}, task="binary")
    assert est.get_params()["probability"] is True


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_suggest_params_returns_nonempty_dict(name):
    study = optuna.create_study()
    trial = study.ask()
    params = models.suggest_params(name, trial)
    assert isinstance(params, dict) and len(params) > 0
