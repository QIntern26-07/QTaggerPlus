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


def test_xgboost_objective_by_task():
    est_binary = models.make_model("xgboost", params={}, task="binary")
    assert est_binary.get_params()["objective"] == "binary:logistic"

    est_multiclass = models.make_model("xgboost", params={}, task="multiclass")
    assert est_multiclass.get_params()["objective"] == "multi:softprob"


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_suggest_params_returns_nonempty_dict(name):
    study = optuna.create_study()
    trial = study.ask()
    params = models.suggest_params(name, trial)
    assert isinstance(params, dict) and len(params) > 0


@pytest.mark.parametrize("name", models.MODEL_NAMES)
def test_suggest_params_integrates_with_make_model(name):
    study = optuna.create_study()
    trial = study.ask()
    params = models.suggest_params(name, trial)
    est = models.make_model(name, params=params, task="binary")
    assert hasattr(est, "fit")
